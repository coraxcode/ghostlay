/*
 * ghostlay.c — Professional floating, click-through image overlay for X11 (Linux)
 *
 * Purpose:
 *   Display an always-on-top reference image with adjustable opacity and size,
 *   allowing the user to trace/copy over it without blocking normal interaction
 *   with underlying applications (click-through).
 *
 * Transparency modes:
 *   1. Compositor (picom/xfwm4/kwin/mutter): ARGB window + XRender premultiplied
 *      alpha — true per-pixel transparency. This is the default when a compositor
 *      is detected.
 *   2. Dither + XShape (-d flag): Ordered dithering simulates opacity by punching
 *      holes (transparent pixels) via XShape bounding mask. Works perfectly on
 *      bare X11 (i3wm, dwm, bspwm, etc.) without any compositor. The image
 *      appears semi-transparent because a percentage of pixels become invisible.
 *   3. Fallback (no compositor, no -d): RGB dimmed by opacity + ShapeBounding
 *      holes for alpha > 0.
 *
 * Click-through:
 *   ShapeInput is set to empty on the visual window so all mouse events pass
 *   through to the application underneath. An invisible InputOnly drag handle
 *   in the top-left corner allows repositioning.
 *
 * Usage:
 *   ghostlay -o 5 -s 500 /path/to/image.png
 *   ghostlay -d -o 3 -s 400 /path/to/image.png    (dither mode for i3wm etc.)
 *
 * Options:
 *   -o N   Opacity 0..10 (10 = fully visible, 0 = almost invisible)
 *   -s PX  Target max dimension in pixels (keeps aspect ratio)
 *   -d     Use dither + XShape transparency (no compositor needed)
 *   -h     Show help
 *
 * Exit:
 *   Ctrl+C, SIGTERM, or kill the process.
 *   If running in background: kill $(pidof ghostlay)
 *
 * Build:
 *   gcc -O2 -std=c11 -Wall -Wextra -Wpedantic ghostlay.c -o ghostlay \
 *     -lX11 -lXext -lXrender -lpng -ljpeg -lm
 */

#define _POSIX_C_SOURCE 200809L

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/shape.h>
#include <X11/extensions/Xrender.h>

#include <png.h>
#include <jpeglib.h>

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <unistd.h>

/* ========================================================================== */
/*  Constants                                                                 */
/* ========================================================================== */

#define GHOSTLAY_VERSION       "2.0.0"
#define MAX_IMAGE_DIM          200000
#define MAX_SCALE_DIM          20000
#define OPACITY_MIN_FACTOR     0.03
#define DRAG_HANDLE_SIZE       48
#define RAISE_INTERVAL_TICKS   4
#define SELECT_TIMEOUT_USEC    250000

/* ========================================================================== */
/*  8x8 Bayer ordered dither matrix (thresholds 0..63)                        */
/* ========================================================================== */

static const uint8_t BAYER8x8[8][8] = {
    {  0, 32,  8, 40,  2, 34, 10, 42 },
    { 48, 16, 56, 24, 50, 18, 58, 26 },
    { 12, 44,  4, 36, 14, 46,  6, 38 },
    { 60, 28, 52, 20, 62, 30, 54, 22 },
    {  3, 35, 11, 43,  1, 33,  9, 41 },
    { 51, 19, 59, 27, 49, 17, 57, 25 },
    { 15, 47,  7, 39, 13, 45,  5, 37 },
    { 63, 31, 55, 23, 61, 29, 53, 21 }
};

/* ========================================================================== */
/*  Image structure                                                           */
/* ========================================================================== */

typedef struct {
    int      w;
    int      h;
    uint8_t *rgba;
} Image;

/* ========================================================================== */
/*  Signal handling: self-pipe for clean shutdown                             */
/* ========================================================================== */

static volatile sig_atomic_t g_running = 1;
static int g_wake_rd = -1;
static int g_wake_wr = -1;

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
    if (g_wake_wr >= 0) {
        const uint8_t b = 0xFF;
        (void)write(g_wake_wr, &b, 1);
    }
}

/* ========================================================================== */
/*  Utility functions                                                         */
/* ========================================================================== */

static void die(const char *msg)
{
    fprintf(stderr, "ghostlay: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void die_errno(const char *msg, const char *detail)
{
    fprintf(stderr, "ghostlay: %s '%s': %s\n", msg, detail, strerror(errno));
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count, size);
    if (!p) die("out of memory");
    return p;
}

static uint8_t clamp_u8(int v)
{
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void image_free(Image *img)
{
    if (!img) return;
    free(img->rgba);
    img->rgba = NULL;
    img->w = 0;
    img->h = 0;
}

/* ========================================================================== */
/*  Endianness and pixel format helpers                                       */
/* ========================================================================== */

static int host_is_little_endian(void)
{
    const uint16_t probe = 0x0001;
    return *(const uint8_t *)&probe == 0x01;
}

static int mask_shift(uint32_t mask)
{
    if (!mask) return 0;
    int s = 0;
    while ((mask & 1u) == 0u) { mask >>= 1; s++; }
    return s;
}

static int mask_bits(uint32_t mask)
{
    int b = 0;
    while (mask) { b += (int)(mask & 1u); mask >>= 1; }
    return b;
}

static uint32_t scale_channel(uint8_t val, int bits)
{
    if (bits <= 0) return 0;
    if (bits >= 8) return (uint32_t)val;
    const uint32_t maxv = (1u << (uint32_t)bits) - 1u;
    return ((uint32_t)val * maxv + 127u) / 255u;
}

typedef struct {
    uint32_t rmask, gmask, bmask, amask;
    int      rsh, gsh, bsh, ash;
    int      rbits, gbits, bbits, abits;
    int      byte_swap;
    int      bpp;
} PixelFormat;

static uint32_t xrender_alpha_mask(const XRenderPictFormat *fmt)
{
    if (!fmt || fmt->type != PictTypeDirect || fmt->direct.alphaMask == 0)
        return 0;
    return (uint32_t)fmt->direct.alphaMask << (uint32_t)fmt->direct.alpha;
}

static PixelFormat pixel_format_from_ximage(const XImage *xi,
                                            const XRenderPictFormat *rfmt)
{
    PixelFormat pf;
    memset(&pf, 0, sizeof(pf));

    pf.rmask = (uint32_t)xi->red_mask;
    pf.gmask = (uint32_t)xi->green_mask;
    pf.bmask = (uint32_t)xi->blue_mask;
    pf.amask = xrender_alpha_mask(rfmt);

    pf.rsh   = mask_shift(pf.rmask);
    pf.gsh   = mask_shift(pf.gmask);
    pf.bsh   = mask_shift(pf.bmask);
    pf.ash   = mask_shift(pf.amask);

    pf.rbits = mask_bits(pf.rmask);
    pf.gbits = mask_bits(pf.gmask);
    pf.bbits = mask_bits(pf.bmask);
    pf.abits = mask_bits(pf.amask);

    pf.bpp = xi->bits_per_pixel / 8;
    if (pf.bpp < 1) pf.bpp = 4;

    int need_swap = 0;
    if (host_is_little_endian()) {
        if (xi->byte_order == MSBFirst) need_swap = 1;
    } else {
        if (xi->byte_order == LSBFirst) need_swap = 1;
    }
    pf.byte_swap = need_swap;

    return pf;
}

static uint32_t pack_pixel(const PixelFormat *pf,
                           uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    uint32_t pix = 0;
    pix |= (scale_channel(r, pf->rbits) << (uint32_t)pf->rsh) & pf->rmask;
    pix |= (scale_channel(g, pf->gbits) << (uint32_t)pf->gsh) & pf->gmask;
    pix |= (scale_channel(b, pf->bbits) << (uint32_t)pf->bsh) & pf->bmask;
    if (pf->amask)
        pix |= (scale_channel(a, pf->abits) << (uint32_t)pf->ash) & pf->amask;

    if (pf->byte_swap) {
        pix = ((pix & 0x000000FFu) << 24) |
              ((pix & 0x0000FF00u) <<  8) |
              ((pix & 0x00FF0000u) >>  8) |
              ((pix & 0xFF000000u) >> 24);
    }
    return pix;
}

static void put_pixel_fast(char *data, int bytes_per_line,
                           int x, int y, uint32_t pix, int bpp)
{
    char *row = data + (size_t)y * (size_t)bytes_per_line;
    switch (bpp) {
    case 4:  ((uint32_t *)(void *)row)[x] = pix; break;
    case 3:
        row[3 * x + 0] = (char)(pix & 0xFF);
        row[3 * x + 1] = (char)((pix >> 8) & 0xFF);
        row[3 * x + 2] = (char)((pix >> 16) & 0xFF);
        break;
    case 2:  ((uint16_t *)(void *)row)[x] = (uint16_t)(pix & 0xFFFF); break;
    default: ((uint32_t *)(void *)row)[x] = pix; break;
    }
}

/* ========================================================================== */
/*  Format detection                                                          */
/* ========================================================================== */

static int file_read_bytes(FILE *f, uint8_t *buf, size_t n)
{
    return fread(buf, 1, n, f) == n;
}

static int detect_png(const char *path)
{
    uint8_t sig[8];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = file_read_bytes(f, sig, 8) && png_sig_cmp(sig, 0, 8) == 0;
    fclose(f);
    return ok;
}

static int detect_jpeg(const char *path)
{
    uint8_t sig[3];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = file_read_bytes(f, sig, 3) &&
             sig[0] == 0xFF && sig[1] == 0xD8 && sig[2] == 0xFF;
    fclose(f);
    return ok;
}

/* ========================================================================== */
/*  PNG loader                                                                */
/* ========================================================================== */

static Image load_png(const char *path)
{
    Image img = {0, 0, NULL};

    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("cannot open PNG", path);

    uint8_t header[8];
    if (!file_read_bytes(fp, header, 8) || png_sig_cmp(header, 0, 8) != 0) {
        fclose(fp);
        die("not a valid PNG file");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                             NULL, NULL, NULL);
    if (!png) { fclose(fp); die("png_create_read_struct failed"); }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        die("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        free(img.rgba);
        die("error reading PNG data");
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = 0, h = 0;
    int bit_depth = 0, color_type = 0;
    png_get_IHDR(png, info, &w, &h, &bit_depth, &color_type, NULL, NULL, NULL);

    if (w == 0 || h == 0 || w > MAX_IMAGE_DIM || h > MAX_IMAGE_DIM) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        die("PNG dimensions invalid or too large");
    }

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    if (!(color_type & PNG_COLOR_MASK_ALPHA))
        png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);

    png_read_update_info(png, info);

    png_size_t rowbytes = png_get_rowbytes(png, info);
    if (rowbytes < (png_size_t)(w * 4)) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        die("unexpected PNG row size");
    }

    img.w = (int)w;
    img.h = (int)h;

    size_t total = (size_t)w * (size_t)h;
    if (total > SIZE_MAX / 4) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        die("image too large to allocate");
    }

    img.rgba = (uint8_t *)xmalloc(total * 4);

    png_bytep *rows = (png_bytep *)xmalloc(sizeof(png_bytep) * h);
    for (png_uint_32 y = 0; y < h; y++)
        rows[y] = img.rgba + (size_t)y * (size_t)w * 4;

    png_read_image(png, rows);
    png_read_end(png, NULL);

    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return img;
}

/* ========================================================================== */
/*  JPEG loader                                                               */
/* ========================================================================== */

struct JpegErrorCtx {
    struct jpeg_error_mgr pub;
    jmp_buf               escape;
};

static void jpeg_error_handler(j_common_ptr cinfo)
{
    struct JpegErrorCtx *ctx = (struct JpegErrorCtx *)cinfo->err;
    longjmp(ctx->escape, 1);
}

static Image load_jpeg(const char *path)
{
    Image img = {0, 0, NULL};

    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("cannot open JPEG", path);

    struct jpeg_decompress_struct cinfo;
    struct JpegErrorCtx jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_handler;

    if (setjmp(jerr.escape)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        free(img.rgba);
        die("error reading JPEG data");
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    img.w = (int)cinfo.output_width;
    img.h = (int)cinfo.output_height;

    if (img.w <= 0 || img.h <= 0 ||
        img.w > MAX_IMAGE_DIM || img.h > MAX_IMAGE_DIM) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        die("invalid JPEG dimensions");
    }

    if (cinfo.output_components != 3) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        die("unexpected JPEG color components");
    }

    size_t total = (size_t)img.w * (size_t)img.h;
    if (total > SIZE_MAX / 4) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        die("image too large to allocate");
    }

    img.rgba = (uint8_t *)xmalloc(total * 4);

    size_t row_stride = (size_t)img.w * 3;
    uint8_t *scanline = (uint8_t *)xmalloc(row_stride);

    while (cinfo.output_scanline < cinfo.output_height) {
        int y = (int)cinfo.output_scanline;
        JSAMPROW row_ptr = scanline;
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);

        uint8_t *dst = img.rgba + (size_t)y * (size_t)img.w * 4;
        for (int x = 0; x < img.w; x++) {
            dst[4 * x + 0] = scanline[3 * x + 0];
            dst[4 * x + 1] = scanline[3 * x + 1];
            dst[4 * x + 2] = scanline[3 * x + 2];
            dst[4 * x + 3] = 255;
        }
    }

    free(scanline);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return img;
}

/* ========================================================================== */
/*  Unified image loader                                                      */
/* ========================================================================== */

static Image load_image(const char *path)
{
    if (detect_png(path))  return load_png(path);
    if (detect_jpeg(path)) return load_jpeg(path);
    fprintf(stderr, "ghostlay: unsupported format (expected PNG or JPEG): %s\n", path);
    exit(EXIT_FAILURE);
}

/* ========================================================================== */
/*  Bilinear scaling                                                          */
/* ========================================================================== */

static Image scale_bilinear(const Image *src, int max_dim)
{
    if (max_dim <= 0 || (src->w <= max_dim && src->h <= max_dim)) {
        Image out;
        out.w = src->w;
        out.h = src->h;
        size_t total = (size_t)src->w * (size_t)src->h * 4;
        out.rgba = (uint8_t *)xmalloc(total);
        memcpy(out.rgba, src->rgba, total);
        return out;
    }

    const int sw = src->w, sh = src->h;
    const int longer = (sw > sh) ? sw : sh;
    double ratio = (double)max_dim / (double)longer;
    int dw = (int)lround((double)sw * ratio);
    int dh = (int)lround((double)sh * ratio);
    if (dw < 1) dw = 1;
    if (dh < 1) dh = 1;

    size_t total = (size_t)dw * (size_t)dh;
    if (total > SIZE_MAX / 4) die("scaled image too large");

    Image dst;
    dst.w = dw;
    dst.h = dh;
    dst.rgba = (uint8_t *)xmalloc(total * 4);

    for (int y = 0; y < dh; y++) {
        double fy = ((double)y + 0.5) * (double)sh / (double)dh - 0.5;
        int y0 = (int)floor(fy), y1 = y0 + 1;
        double wy = fy - (double)y0;
        if (y0 < 0)  { y0 = 0;      wy = 0.0; }
        if (y1 >= sh) { y1 = sh - 1; }

        for (int x = 0; x < dw; x++) {
            double fx = ((double)x + 0.5) * (double)sw / (double)dw - 0.5;
            int x0 = (int)floor(fx), x1 = x0 + 1;
            double wx = fx - (double)x0;
            if (x0 < 0)  { x0 = 0;      wx = 0.0; }
            if (x1 >= sw) { x1 = sw - 1; }

            const uint8_t *p00 = src->rgba + 4 * ((size_t)y0 * sw + x0);
            const uint8_t *p10 = src->rgba + 4 * ((size_t)y0 * sw + x1);
            const uint8_t *p01 = src->rgba + 4 * ((size_t)y1 * sw + x0);
            const uint8_t *p11 = src->rgba + 4 * ((size_t)y1 * sw + x1);
            uint8_t *d = dst.rgba + 4 * ((size_t)y * dw + x);

            for (int c = 0; c < 4; c++) {
                double v = (1.0 - wx) * (1.0 - wy) * (double)p00[c]
                         + wx         * (1.0 - wy) * (double)p10[c]
                         + (1.0 - wx) * wy         * (double)p01[c]
                         + wx         * wy         * (double)p11[c];
                d[c] = clamp_u8((int)lround(v));
            }
        }
    }
    return dst;
}

/* ========================================================================== */
/*  X11 compositor detection                                                  */
/* ========================================================================== */

static int compositor_active(Display *dpy, int screen)
{
    char name[64];
    snprintf(name, sizeof(name), "_NET_WM_CM_S%d", screen);
    Atom sel = XInternAtom(dpy, name, False);
    if (sel == None) return 0;
    return XGetSelectionOwner(dpy, sel) != None;
}

/* ========================================================================== */
/*  X11 ARGB visual selection                                                 */
/* ========================================================================== */

static Visual *find_argb_visual(Display *dpy, int screen, int *out_depth)
{
    XVisualInfo tmpl;
    tmpl.screen = screen;
    tmpl.depth  = 32;
    tmpl.class  = TrueColor;

    int count = 0;
    XVisualInfo *list = XGetVisualInfo(dpy,
        VisualScreenMask | VisualDepthMask | VisualClassMask, &tmpl, &count);
    if (!list || count == 0) {
        if (list) XFree(list);
        return NULL;
    }

    Visual *result = NULL;
    for (int i = 0; i < count; i++) {
        XRenderPictFormat *fmt = XRenderFindVisualFormat(dpy, list[i].visual);
        if (fmt && fmt->type == PictTypeDirect &&
            fmt->direct.alphaMask != 0 && fmt->depth == 32) {
            result = list[i].visual;
            *out_depth = 32;
            break;
        }
    }
    XFree(list);
    return result;
}

/* ========================================================================== */
/*  XShape helpers                                                            */
/* ========================================================================== */

static void shape_input_passthrough(Display *dpy, Window win)
{
    char zero = 0;
    Pixmap pm = XCreateBitmapFromData(dpy, win, &zero, 1, 1);
    if (!pm) return;
    XShapeCombineMask(dpy, win, ShapeInput, 0, 0, pm, ShapeSet);
    XFreePixmap(dpy, pm);
}

static Pixmap shape_mask_from_alpha(Display *dpy, Drawable d,
                                    const uint8_t *rgba, int w, int h)
{
    const int bit_order = BitmapBitOrder(dpy);
    const int stride = (w + 7) / 8;
    size_t sz = (size_t)stride * (size_t)h;
    uint8_t *bits = (uint8_t *)xcalloc(sz, 1);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t alpha = rgba[4 * ((size_t)y * w + x) + 3];
            if (alpha > 0) {
                int byte_idx = y * stride + (x >> 3);
                int bit_idx  = x & 7;
                if (bit_order == LSBFirst)
                    bits[byte_idx] |= (uint8_t)(1u << (unsigned)bit_idx);
                else
                    bits[byte_idx] |= (uint8_t)(1u << (unsigned)(7 - bit_idx));
            }
        }
    }

    Pixmap pm = XCreateBitmapFromData(dpy, d, (const char *)bits,
                                      (unsigned)w, (unsigned)h);
    free(bits);
    return pm;
}

/*
 * Dithered bounding mask: Bayer 8x8 ordered dithering simulates opacity
 * by selectively making pixels visible or invisible via XShape.
 *
 * effective_opacity = (pixel_alpha / 255.0) * global_opacity
 * threshold = (Bayer8x8[y%8][x%8] + 0.5) / 64.0
 * pixel visible if effective_opacity > threshold
 */
static Pixmap shape_mask_dithered(Display *dpy, Drawable d,
                                  const uint8_t *rgba, int w, int h,
                                  double opacity)
{
    const int bit_order = BitmapBitOrder(dpy);
    const int stride = (w + 7) / 8;
    size_t sz = (size_t)stride * (size_t)h;
    uint8_t *bits = (uint8_t *)xcalloc(sz, 1);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t alpha = rgba[4 * ((size_t)y * w + x) + 3];
            double eff = ((double)alpha / 255.0) * opacity;
            double threshold = ((double)BAYER8x8[y & 7][x & 7] + 0.5) / 64.0;

            if (eff > threshold) {
                int byte_idx = y * stride + (x >> 3);
                int bit_idx  = x & 7;
                if (bit_order == LSBFirst)
                    bits[byte_idx] |= (uint8_t)(1u << (unsigned)bit_idx);
                else
                    bits[byte_idx] |= (uint8_t)(1u << (unsigned)(7 - bit_idx));
            }
        }
    }

    Pixmap pm = XCreateBitmapFromData(dpy, d, (const char *)bits,
                                      (unsigned)w, (unsigned)h);
    free(bits);
    return pm;
}

/* ========================================================================== */
/*  X11 overlay context                                                       */
/* ========================================================================== */

typedef struct {
    Display *dpy;
    int      screen;
    Window   root;
    Visual  *visual;
    int      depth;
    Colormap cmap;

    Window   win;
    Window   drag_win;
    Cursor   cursor_move;

    int w, h;
    int drag_handle_size;

    int has_shape;
    int has_render;
    int has_compositor;
    int use_argb;
    int use_dither;

    int dragging;
    int drag_origin_rx, drag_origin_ry;
    int win_origin_x,   win_origin_y;

    XRenderPictFormat *fmt_win;
    Picture            pic_win;
    Pixmap             pm_src;
    Picture            pic_src;
    GC                 gc_pm;

    GC      gc_win;
    XImage *ximg;
} OverlayCtx;

static void overlay_destroy(OverlayCtx *ctx)
{
    if (!ctx || !ctx->dpy) return;

    if (ctx->dragging) {
        XUngrabPointer(ctx->dpy, CurrentTime);
        ctx->dragging = 0;
    }

    if (ctx->ximg)        { XDestroyImage(ctx->ximg);                   ctx->ximg    = NULL; }
    if (ctx->gc_win)      { XFreeGC(ctx->dpy, ctx->gc_win);            ctx->gc_win  = 0;    }
    if (ctx->pic_src)     { XRenderFreePicture(ctx->dpy, ctx->pic_src); ctx->pic_src = 0;    }
    if (ctx->pm_src)      { XFreePixmap(ctx->dpy, ctx->pm_src);        ctx->pm_src  = 0;    }
    if (ctx->pic_win)     { XRenderFreePicture(ctx->dpy, ctx->pic_win); ctx->pic_win = 0;    }
    if (ctx->gc_pm)       { XFreeGC(ctx->dpy, ctx->gc_pm);             ctx->gc_pm   = 0;    }
    if (ctx->cursor_move) { XFreeCursor(ctx->dpy, ctx->cursor_move);    ctx->cursor_move = 0;}
    if (ctx->drag_win)    { XDestroyWindow(ctx->dpy, ctx->drag_win);    ctx->drag_win = 0;   }
    if (ctx->win)         { XDestroyWindow(ctx->dpy, ctx->win);         ctx->win      = 0;   }
    if (ctx->cmap)        { XFreeColormap(ctx->dpy, ctx->cmap);         ctx->cmap     = 0;   }

    XCloseDisplay(ctx->dpy);
    ctx->dpy = NULL;
}

static void overlay_get_position(OverlayCtx *ctx, int *ox, int *oy)
{
    Window child;
    int x = 0, y = 0;
    XTranslateCoordinates(ctx->dpy, ctx->win, ctx->root, 0, 0, &x, &y, &child);
    if (ox) *ox = x;
    if (oy) *oy = y;
}

static void overlay_move(OverlayCtx *ctx, int x, int y)
{
    XMoveWindow(ctx->dpy, ctx->win,      x, y);
    XMoveWindow(ctx->dpy, ctx->drag_win, x, y);
}

static void overlay_set_hints(OverlayCtx *ctx)
{
    Atom wm_type = XInternAtom(ctx->dpy, "_NET_WM_WINDOW_TYPE", False);
    Atom type_util = XInternAtom(ctx->dpy, "_NET_WM_WINDOW_TYPE_UTILITY", False);
    if (wm_type != None && type_util != None)
        XChangeProperty(ctx->dpy, ctx->win, wm_type, XA_ATOM, 32,
                        PropModeReplace, (unsigned char *)&type_util, 1);

    Atom wm_state = XInternAtom(ctx->dpy, "_NET_WM_STATE", False);
    Atom state_above = XInternAtom(ctx->dpy, "_NET_WM_STATE_ABOVE", False);
    if (wm_state != None && state_above != None)
        XChangeProperty(ctx->dpy, ctx->win, wm_state, XA_ATOM, 32,
                        PropModeReplace, (unsigned char *)&state_above, 1);

    XClassHint class_hint;
    class_hint.res_name  = (char *)"ghostlay";
    class_hint.res_class = (char *)"Ghostlay";
    XSetClassHint(ctx->dpy, ctx->win, &class_hint);
}

static void overlay_init(OverlayCtx *ctx, int w, int h, int use_dither)
{
    memset(ctx, 0, sizeof(*ctx));

    ctx->dpy = XOpenDisplay(NULL);
    if (!ctx->dpy) die("cannot open X display (is $DISPLAY set?)");

    ctx->screen = DefaultScreen(ctx->dpy);
    ctx->root   = RootWindow(ctx->dpy, ctx->screen);
    ctx->w = w;  ctx->h = h;
    ctx->drag_handle_size = DRAG_HANDLE_SIZE;
    ctx->use_dither = use_dither;

    int ev = 0, err = 0;
    ctx->has_shape  = XShapeQueryExtension(ctx->dpy, &ev, &err);
    ctx->has_render = XRenderQueryExtension(ctx->dpy, &ev, &err);
    ctx->has_compositor = compositor_active(ctx->dpy, ctx->screen);

    Visual *vis = NULL;
    int depth = 0;
    if (!use_dither && ctx->has_render)
        vis = find_argb_visual(ctx->dpy, ctx->screen, &depth);

    if (vis) {
        ctx->visual = vis;
        ctx->depth  = depth;
    } else {
        ctx->visual = DefaultVisual(ctx->dpy, ctx->screen);
        ctx->depth  = DefaultDepth(ctx->dpy, ctx->screen);
    }

    ctx->use_argb = (!use_dither && ctx->has_render && ctx->depth == 32);
    ctx->cmap = XCreateColormap(ctx->dpy, ctx->root, ctx->visual, AllocNone);

    int scr_w = DisplayWidth(ctx->dpy, ctx->screen);
    int scr_h = DisplayHeight(ctx->dpy, ctx->screen);
    int pos_x = (scr_w - w) / 2;
    int pos_y = (scr_h - h) / 2;
    if (pos_x < 0) pos_x = 0;
    if (pos_y < 0) pos_y = 0;

    XSetWindowAttributes attr;
    memset(&attr, 0, sizeof(attr));
    attr.colormap         = ctx->cmap;
    attr.border_pixel     = 0;
    attr.background_pixel = 0;
    attr.override_redirect = True;
    attr.event_mask        = ExposureMask | StructureNotifyMask;

    ctx->win = XCreateWindow(ctx->dpy, ctx->root,
        pos_x, pos_y, (unsigned)w, (unsigned)h,
        0, ctx->depth, InputOutput, ctx->visual,
        CWColormap | CWBorderPixel | CWBackPixel | CWOverrideRedirect | CWEventMask,
        &attr);
    if (!ctx->win) die("failed to create overlay window");

    XStoreName(ctx->dpy, ctx->win, "ghostlay");
    overlay_set_hints(ctx);

    XSetWindowAttributes dattr;
    memset(&dattr, 0, sizeof(dattr));
    dattr.override_redirect = True;
    dattr.event_mask = ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
    int hw = (ctx->drag_handle_size < w) ? ctx->drag_handle_size : w;
    int hh = (ctx->drag_handle_size < h) ? ctx->drag_handle_size : h;

    ctx->drag_win = XCreateWindow(ctx->dpy, ctx->root,
        pos_x, pos_y, (unsigned)hw, (unsigned)hh,
        0, 0, InputOnly, CopyFromParent,
        CWOverrideRedirect | CWEventMask, &dattr);
    if (!ctx->drag_win) die("failed to create drag handle window");

    ctx->cursor_move = XCreateFontCursor(ctx->dpy, XC_fleur);
    XDefineCursor(ctx->dpy, ctx->drag_win, ctx->cursor_move);

    if (ctx->has_shape) {
        shape_input_passthrough(ctx->dpy, ctx->win);
    } else {
        fprintf(stderr, "ghostlay: warning: XShape unavailable, click-through disabled\n");
    }

    if (use_dither)
        fprintf(stderr, "ghostlay: dither + XShape transparency enabled\n");
    else if (!ctx->has_compositor)
        fprintf(stderr, "ghostlay: no compositor — use -d for dithered transparency, or run picom\n");

    if (ctx->use_argb) {
        ctx->fmt_win = XRenderFindVisualFormat(ctx->dpy, ctx->visual);
        if (!ctx->fmt_win) die("XRenderFindVisualFormat failed");
        XRenderPictureAttributes pa;
        memset(&pa, 0, sizeof(pa));
        ctx->pic_win = XRenderCreatePicture(ctx->dpy, ctx->win, ctx->fmt_win, 0, &pa);
        if (!ctx->pic_win) die("XRenderCreatePicture failed");
    } else {
        ctx->gc_win = XCreateGC(ctx->dpy, ctx->win, 0, NULL);
        if (!ctx->gc_win) die("XCreateGC failed");
    }

    XMapRaised(ctx->dpy, ctx->win);
    XMapRaised(ctx->dpy, ctx->drag_win);
    XFlush(ctx->dpy);
}

/* ========================================================================== */
/*  Drawing: XRender ARGB path                                                */
/* ========================================================================== */

static void overlay_upload_argb(OverlayCtx *ctx, const uint8_t *rgba, double opacity)
{
    if (!ctx->use_argb) return;
    if (opacity < OPACITY_MIN_FACTOR) opacity = OPACITY_MIN_FACTOR;
    if (opacity > 1.0) opacity = 1.0;

    if (ctx->pic_src) { XRenderFreePicture(ctx->dpy, ctx->pic_src); ctx->pic_src = 0; }
    if (ctx->pm_src)  { XFreePixmap(ctx->dpy, ctx->pm_src);         ctx->pm_src  = 0; }

    ctx->pm_src = XCreatePixmap(ctx->dpy, ctx->win,
                                (unsigned)ctx->w, (unsigned)ctx->h, 32);
    if (!ctx->pm_src) die("XCreatePixmap failed");

    if (!ctx->gc_pm) {
        ctx->gc_pm = XCreateGC(ctx->dpy, ctx->pm_src, 0, NULL);
        if (!ctx->gc_pm) die("XCreateGC failed for pixmap");
    }

    XImage *xi = XCreateImage(ctx->dpy, ctx->visual, 32, ZPixmap, 0, NULL,
                              (unsigned)ctx->w, (unsigned)ctx->h, 32, 0);
    if (!xi) die("XCreateImage failed");

    size_t datasz = (size_t)xi->bytes_per_line * (size_t)xi->height;
    xi->data = (char *)xmalloc(datasz);
    memset(xi->data, 0, datasz);

    PixelFormat pf = pixel_format_from_ximage(xi, ctx->fmt_win);
    const int use_alpha = ctx->has_compositor;

    for (int y = 0; y < ctx->h; y++) {
        for (int x = 0; x < ctx->w; x++) {
            const uint8_t *p = rgba + 4 * ((size_t)y * ctx->w + x);
            uint8_t sr = p[0], sg = p[1], sb = p[2], sa = p[3];
            uint8_t a8, r8, g8, b8;

            if (use_alpha) {
                int a_eff = (int)lround((double)sa * opacity);
                a8 = clamp_u8(a_eff);
                r8 = (uint8_t)(((unsigned)sr * a8 + 127u) / 255u);
                g8 = (uint8_t)(((unsigned)sg * a8 + 127u) / 255u);
                b8 = (uint8_t)(((unsigned)sb * a8 + 127u) / 255u);
            } else {
                a8 = 255;
                r8 = clamp_u8((int)lround((double)sr * opacity));
                g8 = clamp_u8((int)lround((double)sg * opacity));
                b8 = clamp_u8((int)lround((double)sb * opacity));
            }

            uint32_t pix = pack_pixel(&pf, r8, g8, b8, a8);
            put_pixel_fast(xi->data, xi->bytes_per_line, x, y, pix, pf.bpp);
        }
    }

    XPutImage(ctx->dpy, ctx->pm_src, ctx->gc_pm, xi, 0, 0, 0, 0,
              (unsigned)ctx->w, (unsigned)ctx->h);
    XDestroyImage(xi);

    XRenderPictFormat *fmt_src = XRenderFindStandardFormat(ctx->dpy, PictStandardARGB32);
    if (!fmt_src) fmt_src = ctx->fmt_win;

    XRenderPictureAttributes pa;
    memset(&pa, 0, sizeof(pa));
    ctx->pic_src = XRenderCreatePicture(ctx->dpy, ctx->pm_src, fmt_src, 0, &pa);
    if (!ctx->pic_src) die("XRenderCreatePicture failed for source");
}

static void overlay_draw_argb(OverlayCtx *ctx)
{
    if (!ctx->use_argb || !ctx->pic_src) return;
    XRenderColor clear = {0, 0, 0, 0};
    XRenderFillRectangle(ctx->dpy, PictOpClear, ctx->pic_win, &clear,
                         0, 0, (unsigned)ctx->w, (unsigned)ctx->h);
    XRenderComposite(ctx->dpy, PictOpSrc, ctx->pic_src, None, ctx->pic_win,
                     0, 0, 0, 0, 0, 0, (unsigned)ctx->w, (unsigned)ctx->h);
}

/* ========================================================================== */
/*  Drawing: non-ARGB path (fallback and dither)                              */
/* ========================================================================== */

static void overlay_build_ximage(OverlayCtx *ctx, const uint8_t *rgba, double opacity)
{
    if (ctx->use_argb) return;
    if (opacity < OPACITY_MIN_FACTOR) opacity = OPACITY_MIN_FACTOR;
    if (opacity > 1.0) opacity = 1.0;

    if (ctx->ximg) { XDestroyImage(ctx->ximg); ctx->ximg = NULL; }

    XImage *xi = XCreateImage(ctx->dpy, ctx->visual, (unsigned)ctx->depth,
                              ZPixmap, 0, NULL,
                              (unsigned)ctx->w, (unsigned)ctx->h, 32, 0);
    if (!xi) die("XCreateImage failed");

    size_t datasz = (size_t)xi->bytes_per_line * (size_t)xi->height;
    xi->data = (char *)xmalloc(datasz);
    memset(xi->data, 0, datasz);

    PixelFormat pf = pixel_format_from_ximage(xi, NULL);
    const int is_dither = ctx->use_dither;

    for (int y = 0; y < ctx->h; y++) {
        for (int x = 0; x < ctx->w; x++) {
            const uint8_t *p = rgba + 4 * ((size_t)y * ctx->w + x);
            uint8_t r, g, b;

            if (is_dither) {
                r = p[0]; g = p[1]; b = p[2];
            } else {
                r = clamp_u8((int)lround((double)p[0] * opacity));
                g = clamp_u8((int)lround((double)p[1] * opacity));
                b = clamp_u8((int)lround((double)p[2] * opacity));
            }

            uint32_t pix = pack_pixel(&pf, r, g, b, 255);
            put_pixel_fast(xi->data, xi->bytes_per_line, x, y, pix, pf.bpp);
        }
    }
    ctx->ximg = xi;
}

static void overlay_draw_ximage(OverlayCtx *ctx)
{
    if (ctx->use_argb || !ctx->ximg) return;
    XPutImage(ctx->dpy, ctx->win, ctx->gc_win, ctx->ximg,
              0, 0, 0, 0, (unsigned)ctx->w, (unsigned)ctx->h);
}

/* ========================================================================== */
/*  Unified redraw                                                            */
/* ========================================================================== */

static void overlay_redraw(OverlayCtx *ctx)
{
    if (ctx->use_argb) overlay_draw_argb(ctx);
    else               overlay_draw_ximage(ctx);
    XRaiseWindow(ctx->dpy, ctx->win);
    XRaiseWindow(ctx->dpy, ctx->drag_win);
    XFlush(ctx->dpy);
}

static void overlay_apply_bounding_shape(OverlayCtx *ctx,
                                         const uint8_t *rgba, double opacity)
{
    if (!ctx->has_shape) return;
    Pixmap pm;
    if (ctx->use_dither)
        pm = shape_mask_dithered(ctx->dpy, ctx->win, rgba, ctx->w, ctx->h, opacity);
    else
        pm = shape_mask_from_alpha(ctx->dpy, ctx->win, rgba, ctx->w, ctx->h);
    if (!pm) return;
    XShapeCombineMask(ctx->dpy, ctx->win, ShapeBounding, 0, 0, pm, ShapeSet);
    XFreePixmap(ctx->dpy, pm);
}

/* ========================================================================== */
/*  Self-pipe drain                                                           */
/* ========================================================================== */

static void drain_wake_pipe(void)
{
    if (g_wake_rd < 0) return;
    uint8_t buf[64];
    while (read(g_wake_rd, buf, sizeof(buf)) > 0) ;
}

/* ========================================================================== */
/*  Event loop                                                                */
/* ========================================================================== */

static void overlay_loop(OverlayCtx *ctx)
{
    const int xfd = ConnectionNumber(ctx->dpy);
    int raise_counter = 0;

    while (g_running) {
        while (XPending(ctx->dpy) > 0) {
            XEvent ev;
            XNextEvent(ctx->dpy, &ev);

            switch (ev.type) {
            case Expose:
                if (ev.xexpose.count == 0) overlay_redraw(ctx);
                break;
            case ButtonPress:
                if (ev.xbutton.window == ctx->drag_win && ev.xbutton.button == Button1) {
                    ctx->dragging = 1;
                    ctx->drag_origin_rx = ev.xbutton.x_root;
                    ctx->drag_origin_ry = ev.xbutton.y_root;
                    overlay_get_position(ctx, &ctx->win_origin_x, &ctx->win_origin_y);
                    XGrabPointer(ctx->dpy, ctx->drag_win, False,
                                 PointerMotionMask | ButtonReleaseMask,
                                 GrabModeAsync, GrabModeAsync,
                                 None, ctx->cursor_move, CurrentTime);
                }
                break;
            case MotionNotify:
                if (ctx->dragging) {
                    XEvent latest = ev, tmp;
                    while (XCheckTypedWindowEvent(ctx->dpy, ctx->drag_win, MotionNotify, &tmp))
                        latest = tmp;
                    int dx = latest.xmotion.x_root - ctx->drag_origin_rx;
                    int dy = latest.xmotion.y_root - ctx->drag_origin_ry;
                    overlay_move(ctx, ctx->win_origin_x + dx, ctx->win_origin_y + dy);
                    XRaiseWindow(ctx->dpy, ctx->win);
                    XRaiseWindow(ctx->dpy, ctx->drag_win);
                    XFlush(ctx->dpy);
                }
                break;
            case ButtonRelease:
                if (ev.xbutton.window == ctx->drag_win && ev.xbutton.button == Button1) {
                    ctx->dragging = 0;
                    XUngrabPointer(ctx->dpy, CurrentTime);
                }
                break;
            default:
                break;
            }
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(xfd, &rfds);
        int maxfd = xfd;
        if (g_wake_rd >= 0) {
            FD_SET(g_wake_rd, &rfds);
            if (g_wake_rd > maxfd) maxfd = g_wake_rd;
        }

        struct timeval tv;
        tv.tv_sec  = 0;
        tv.tv_usec = SELECT_TIMEOUT_USEC;

        int ret = select(maxfd + 1, &rfds, NULL, NULL, &tv);
        if (ret < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "ghostlay: select error: %s\n", strerror(errno));
            break;
        }

        if (g_wake_rd >= 0 && FD_ISSET(g_wake_rd, &rfds)) {
            drain_wake_pipe();
            break;
        }

        if (!g_running) break;

        raise_counter++;
        if (raise_counter >= RAISE_INTERVAL_TICKS) {
            raise_counter = 0;
            XRaiseWindow(ctx->dpy, ctx->win);
            XRaiseWindow(ctx->dpy, ctx->drag_win);
            XFlush(ctx->dpy);
        }
    }
}

/* ========================================================================== */
/*  Usage                                                                     */
/* ========================================================================== */

static void print_usage(FILE *out)
{
    fprintf(out,
        "ghostlay v%s — floating click-through image overlay for X11\n"
        "\n"
        "Usage:\n"
        "  ghostlay [-o 0..10] [-s PX] [-d] <image.(png|jpg|jpeg)>\n"
        "\n"
        "Options:\n"
        "  -o N   Opacity level 0..10 (10 = fully visible, 0 = near invisible)\n"
        "  -s PX  Scale to max dimension PX pixels (preserves aspect ratio)\n"
        "  -d     Dither transparency (works without compositor, ideal for i3wm)\n"
        "  -h     Show this help\n"
        "\n"
        "Controls:\n"
        "  Drag the top-left corner to reposition the overlay.\n"
        "\n"
        "Exit:\n"
        "  Ctrl+C, SIGTERM, or: kill $(pidof ghostlay)\n"
        "\n"
        "Examples:\n"
        "  ghostlay -o 5 -s 500 reference.png           (with compositor)\n"
        "  ghostlay -d -o 4 -s 400 reference.png        (i3wm, no compositor)\n"
        "  ghostlay -d -o 3 -s 300 sketch.jpg &         (background mode)\n"
        "\n"
        "Notes:\n"
        "  With compositor (picom, xfwm4, kwin, mutter): true alpha blending.\n"
        "  With -d: ordered dithering via XShape (no compositor needed).\n"
        "  Without either: fallback dimmed rendering with shape holes.\n",
        GHOSTLAY_VERSION
    );
}

/* ========================================================================== */
/*  PID file                                                                  */
/* ========================================================================== */

static char g_pid_path[512] = {0};

static void write_pid_file(void)
{
    const char *runtime = getenv("XDG_RUNTIME_DIR");
    if (!runtime) runtime = "/tmp";
    snprintf(g_pid_path, sizeof(g_pid_path), "%s/ghostlay.pid", runtime);
    FILE *f = fopen(g_pid_path, "w");
    if (f) { fprintf(f, "%d\n", (int)getpid()); fclose(f); }
}

static void remove_pid_file(void)
{
    if (g_pid_path[0]) unlink(g_pid_path);
}

/* ========================================================================== */
/*  Main                                                                      */
/* ========================================================================== */

int main(int argc, char **argv)
{
    int opacity_level = 10;
    int target_size   = 0;
    int use_dither    = 0;
    int opt;

    while ((opt = getopt(argc, argv, "o:s:dh")) != -1) {
        switch (opt) {
        case 'o': {
            char *end = NULL;
            long v = strtol(optarg, &end, 10);
            if (!end || *end != '\0') {
                fprintf(stderr, "ghostlay: invalid opacity value '%s'\n", optarg);
                return 2;
            }
            if (v < 0)  v = 0;
            if (v > 10) v = 10;
            opacity_level = (int)v;
            break;
        }
        case 's': {
            char *end = NULL;
            long v = strtol(optarg, &end, 10);
            if (!end || *end != '\0' || v <= 0) {
                fprintf(stderr, "ghostlay: invalid size value '%s'\n", optarg);
                return 2;
            }
            if (v > MAX_SCALE_DIM) v = MAX_SCALE_DIM;
            target_size = (int)v;
            break;
        }
        case 'd':
            use_dither = 1;
            break;
        case 'h':
            print_usage(stdout);
            return 0;
        default:
            print_usage(stderr);
            return 2;
        }
    }

    if (optind >= argc) {
        print_usage(stderr);
        return 2;
    }

    const char *image_path = argv[optind];

    if (access(image_path, R_OK) != 0) {
        fprintf(stderr, "ghostlay: cannot access '%s': %s\n",
                image_path, strerror(errno));
        return 1;
    }

    write_pid_file();

    int pfd[2];
    if (pipe(pfd) == 0) {
        g_wake_rd = pfd[0];
        g_wake_wr = pfd[1];
        fcntl(g_wake_rd, F_SETFL, O_NONBLOCK);
        fcntl(g_wake_wr, F_SETFL, O_NONBLOCK);
        fcntl(g_wake_rd, F_SETFD, FD_CLOEXEC);
        fcntl(g_wake_wr, F_SETFD, FD_CLOEXEC);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP,  &sa, NULL);

    Image src    = load_image(image_path);
    Image scaled = scale_bilinear(&src, target_size);

    fprintf(stderr, "ghostlay: loaded %dx%d -> %dx%d, opacity %d/10%s (pid %d)\n",
            src.w, src.h, scaled.w, scaled.h, opacity_level,
            use_dither ? ", dither" : "", (int)getpid());

    double opacity = (opacity_level == 0) ? OPACITY_MIN_FACTOR
                                          : (double)opacity_level / 10.0;

    OverlayCtx ctx;
    overlay_init(&ctx, scaled.w, scaled.h, use_dither);

    if (ctx.use_argb) {
        overlay_upload_argb(&ctx, scaled.rgba, opacity);
        if (!ctx.has_compositor)
            overlay_apply_bounding_shape(&ctx, scaled.rgba, opacity);
    } else {
        overlay_build_ximage(&ctx, scaled.rgba, opacity);
        overlay_apply_bounding_shape(&ctx, scaled.rgba, opacity);
    }

    overlay_redraw(&ctx);
    overlay_loop(&ctx);

    overlay_destroy(&ctx);
    image_free(&src);
    image_free(&scaled);

    if (g_wake_rd >= 0) close(g_wake_rd);
    if (g_wake_wr >= 0) close(g_wake_wr);

    remove_pid_file();
    fprintf(stderr, "ghostlay: exiting cleanly\n");
    return 0;
}
