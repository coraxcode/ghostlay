/*
 * ghostlay.c v4.0 — click-through image overlay for X11
 *
 * Modes: compositor ARGB | dither+XShape (-d) | fallback dimmed
 * Controls: Alt+Drag=move, Alt+Plus/Minus=resize, Alt+Q=quit
 *
 * gcc -O2 -std=c11 -Wall -Wextra -Wpedantic ghostlay.c -o ghostlay \
 *   -lX11 -lXext -lXrender -lpng -ljpeg -lm
 */

#define _POSIX_C_SOURCE 200809L

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
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
#include <sys/prctl.h>
#include <sys/select.h>
#include <unistd.h>

#define VERSION       "4.0.0"
#define MAX_IMG_DIM   200000
#define MAX_SCALE_DIM 20000
#define MIN_DIM       20
#define OPACITY_MIN   0.03
#define RAISE_TICKS   4
#define SELECT_US     250000
#define RESIZE_STEP   40

static const uint8_t BAYER8[8][8] = {
    { 0,32, 8,40, 2,34,10,42}, {48,16,56,24,50,18,58,26},
    {12,44, 4,36,14,46, 6,38}, {60,28,52,20,62,30,54,22},
    { 3,35,11,43, 1,33, 9,41}, {51,19,59,27,49,17,57,25},
    {15,47, 7,39,13,45, 5,37}, {63,31,55,23,61,29,53,21}
};

typedef struct {
    int w, h;
    uint8_t *rgba;
} Image;

/* --- signals & X error --- */

static volatile sig_atomic_t g_running = 1;
static int g_wake_rd = -1;
static int g_wake_wr = -1;

static void sig_handler(int s)
{
    (void)s;
    g_running = 0;
    if (g_wake_wr >= 0) {
        ssize_t r = write(g_wake_wr, "\xff", 1);
        (void)r;
    }
}

static int xio_error(Display *d)
{
    (void)d;
    fprintf(stderr, "ghostlay: X connection lost\n");
    _exit(0);
    return 0;
}

static int x_error(Display *d, XErrorEvent *e)
{
    (void)d;
    (void)e;
    return 0;
}

/* --- util --- */

static void die(const char *msg)
{
    fprintf(stderr, "ghostlay: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t s)
{
    void *p = calloc(n, s);
    if (!p) die("out of memory");
    return p;
}

static uint8_t clamp8(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void image_free(Image *img)
{
    if (img) {
        free(img->rgba);
        img->rgba = NULL;
        img->w = 0;
        img->h = 0;
    }
}

/* --- pixel format --- */

static int host_le(void)
{
    uint16_t x = 1;
    return *(uint8_t *)&x;
}

static int mshift(uint32_t m)
{
    if (!m) return 0;
    int s = 0;
    while (!(m & 1)) { m >>= 1; s++; }
    return s;
}

static int mbits(uint32_t m)
{
    int b = 0;
    while (m) { b += m & 1; m >>= 1; }
    return b;
}

static uint32_t schan(uint8_t v, int b)
{
    if (b <= 0) return 0;
    if (b >= 8) return v;
    return ((uint32_t)v * ((1u << b) - 1) + 127) / 255;
}

typedef struct {
    uint32_t rm, gm, bm, am;
    int rs, gs, bs, as;
    int rb, gb, bb, ab;
    int swap, bpp;
} PxFmt;

static uint32_t xr_amask(const XRenderPictFormat *f)
{
    if (!f || f->type != PictTypeDirect || !f->direct.alphaMask) return 0;
    return (uint32_t)f->direct.alphaMask << f->direct.alpha;
}

static PxFmt make_pxfmt(const XImage *xi, const XRenderPictFormat *rf)
{
    PxFmt p;
    memset(&p, 0, sizeof(p));
    p.rm = (uint32_t)xi->red_mask;
    p.gm = (uint32_t)xi->green_mask;
    p.bm = (uint32_t)xi->blue_mask;
    p.am = xr_amask(rf);
    p.rs = mshift(p.rm);
    p.gs = mshift(p.gm);
    p.bs = mshift(p.bm);
    p.as = mshift(p.am);
    p.rb = mbits(p.rm);
    p.gb = mbits(p.gm);
    p.bb = mbits(p.bm);
    p.ab = mbits(p.am);
    p.bpp = xi->bits_per_pixel / 8;
    if (p.bpp < 1) p.bpp = 4;
    p.swap = host_le() ? (xi->byte_order == MSBFirst) : (xi->byte_order == LSBFirst);
    return p;
}

static uint32_t packpx(const PxFmt *p, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    uint32_t px = 0;
    px |= (schan(r, p->rb) << p->rs) & p->rm;
    px |= (schan(g, p->gb) << p->gs) & p->gm;
    px |= (schan(b, p->bb) << p->bs) & p->bm;
    if (p->am)
        px |= (schan(a, p->ab) << p->as) & p->am;
    if (p->swap)
        px = (px >> 24) | ((px >> 8) & 0xFF00) | ((px << 8) & 0xFF0000) | (px << 24);
    return px;
}

static void putpx(char *data, int bpl, int x, int y, uint32_t px, int bpp)
{
    char *row = data + (size_t)y * bpl;
    switch (bpp) {
    case 4:
        ((uint32_t *)(void *)row)[x] = px;
        break;
    case 2:
        ((uint16_t *)(void *)row)[x] = (uint16_t)px;
        break;
    case 3:
        row[3 * x + 0] = (char)(px & 0xFF);
        row[3 * x + 1] = (char)((px >> 8) & 0xFF);
        row[3 * x + 2] = (char)((px >> 16) & 0xFF);
        break;
    default:
        ((uint32_t *)(void *)row)[x] = px;
        break;
    }
}

/* --- format detect --- */

static int freadn(FILE *f, uint8_t *b, size_t n)
{
    return fread(b, 1, n, f) == n;
}

static int is_png(const char *path)
{
    uint8_t s[8];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = freadn(f, s, 8) && !png_sig_cmp(s, 0, 8);
    fclose(f);
    return ok;
}

static int is_jpeg(const char *path)
{
    uint8_t s[3];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = freadn(f, s, 3) && s[0] == 0xFF && s[1] == 0xD8 && s[2] == 0xFF;
    fclose(f);
    return ok;
}

/* --- PNG loader --- */

static Image load_png(const char *path)
{
    Image img = {0, 0, NULL};
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ghostlay: %s: %s\n", path, strerror(errno));
        exit(1);
    }

    uint8_t hdr[8];
    if (!freadn(fp, hdr, 8) || png_sig_cmp(hdr, 0, 8)) {
        fclose(fp);
        die("not a valid PNG");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); die("png init failed"); }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        die("png info failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        free(img.rgba);
        die("PNG read error");
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    png_uint_32 w = 0, h = 0;
    int bd = 0, ct = 0;
    png_get_IHDR(png, info, &w, &h, &bd, &ct, NULL, NULL, NULL);
    if (!w || !h || w > MAX_IMG_DIM || h > MAX_IMG_DIM) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        die("PNG too large");
    }

    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (!(ct & PNG_COLOR_MASK_ALPHA)) png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    img.w = (int)w;
    img.h = (int)h;
    size_t tot = (size_t)w * h;
    if (tot > SIZE_MAX / 4) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        die("too large");
    }
    img.rgba = (uint8_t *)xmalloc(tot * 4);

    png_bytep *rows = (png_bytep *)xmalloc(sizeof(png_bytep) * h);
    for (png_uint_32 y = 0; y < h; y++)
        rows[y] = img.rgba + (size_t)y * w * 4;
    png_read_image(png, rows);
    png_read_end(png, NULL);

    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return img;
}

/* --- JPEG loader --- */

struct JErr {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
};

static void jerr_exit(j_common_ptr c)
{
    longjmp(((struct JErr *)c->err)->jb, 1);
}

static Image load_jpeg(const char *path)
{
    Image img = {0, 0, NULL};
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ghostlay: %s: %s\n", path, strerror(errno));
        exit(1);
    }

    struct jpeg_decompress_struct ci;
    struct JErr je;
    ci.err = jpeg_std_error(&je.pub);
    je.pub.error_exit = jerr_exit;

    if (setjmp(je.jb)) {
        jpeg_destroy_decompress(&ci);
        fclose(fp);
        free(img.rgba);
        die("JPEG error");
    }

    jpeg_create_decompress(&ci);
    jpeg_stdio_src(&ci, fp);
    jpeg_read_header(&ci, TRUE);
    ci.out_color_space = JCS_RGB;
    jpeg_start_decompress(&ci);

    img.w = (int)ci.output_width;
    img.h = (int)ci.output_height;
    if (img.w <= 0 || img.h <= 0 || img.w > MAX_IMG_DIM || img.h > MAX_IMG_DIM) {
        jpeg_finish_decompress(&ci);
        jpeg_destroy_decompress(&ci);
        fclose(fp);
        die("JPEG too large");
    }
    if (ci.output_components != 3) {
        jpeg_finish_decompress(&ci);
        jpeg_destroy_decompress(&ci);
        fclose(fp);
        die("unexpected JPEG components");
    }

    size_t tot = (size_t)img.w * img.h;
    if (tot > SIZE_MAX / 4) {
        jpeg_finish_decompress(&ci);
        jpeg_destroy_decompress(&ci);
        fclose(fp);
        die("too large");
    }
    img.rgba = (uint8_t *)xmalloc(tot * 4);
    uint8_t *sl = (uint8_t *)xmalloc((size_t)img.w * 3);

    while (ci.output_scanline < ci.output_height) {
        int y = (int)ci.output_scanline;
        JSAMPROW rp = sl;
        jpeg_read_scanlines(&ci, &rp, 1);
        uint8_t *d = img.rgba + (size_t)y * img.w * 4;
        for (int x = 0; x < img.w; x++) {
            d[4 * x + 0] = sl[3 * x + 0];
            d[4 * x + 1] = sl[3 * x + 1];
            d[4 * x + 2] = sl[3 * x + 2];
            d[4 * x + 3] = 255;
        }
    }

    free(sl);
    jpeg_finish_decompress(&ci);
    jpeg_destroy_decompress(&ci);
    fclose(fp);
    return img;
}

static Image load_image(const char *path)
{
    if (is_png(path)) return load_png(path);
    if (is_jpeg(path)) return load_jpeg(path);
    fprintf(stderr, "ghostlay: unsupported format: %s\n", path);
    exit(1);
}

/* --- bilinear scale --- */

static Image scale_bilinear(const Image *src, int maxd)
{
    /*
     * Scale so that the longer side equals maxd.
     * Always scale, even if image is smaller (upscale).
     * maxd <= 0 means no scaling.
     */
    if (maxd <= 0) {
        Image o;
        o.w = src->w;
        o.h = src->h;
        size_t t = (size_t)o.w * o.h * 4;
        o.rgba = (uint8_t *)xmalloc(t);
        memcpy(o.rgba, src->rgba, t);
        return o;
    }

    int sw = src->w, sh = src->h;
    int longer = sw > sh ? sw : sh;
    double ratio = (double)maxd / (double)longer;
    int dw = (int)lround(sw * ratio);
    int dh = (int)lround(sh * ratio);
    if (dw < 1) dw = 1;
    if (dh < 1) dh = 1;

    /* If ratio is ~1.0 and dimensions match, just copy */
    if (dw == sw && dh == sh) {
        Image o;
        o.w = sw;
        o.h = sh;
        size_t t = (size_t)sw * sh * 4;
        o.rgba = (uint8_t *)xmalloc(t);
        memcpy(o.rgba, src->rgba, t);
        return o;
    }

    size_t tot = (size_t)dw * dh;
    if (tot > SIZE_MAX / 4) die("scale too large");

    Image dst;
    dst.w = dw;
    dst.h = dh;
    dst.rgba = (uint8_t *)xmalloc(tot * 4);

    for (int y = 0; y < dh; y++) {
        double fy = ((double)y + 0.5) * sh / dh - 0.5;
        int y0 = (int)floor(fy), y1 = y0 + 1;
        double wy = fy - y0;
        if (y0 < 0) { y0 = 0; wy = 0; }
        if (y1 >= sh) y1 = sh - 1;

        for (int x = 0; x < dw; x++) {
            double fx = ((double)x + 0.5) * sw / dw - 0.5;
            int x0 = (int)floor(fx), x1 = x0 + 1;
            double wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x1 >= sw) x1 = sw - 1;

            const uint8_t *p00 = src->rgba + 4 * ((size_t)y0 * sw + x0);
            const uint8_t *p10 = src->rgba + 4 * ((size_t)y0 * sw + x1);
            const uint8_t *p01 = src->rgba + 4 * ((size_t)y1 * sw + x0);
            const uint8_t *p11 = src->rgba + 4 * ((size_t)y1 * sw + x1);
            uint8_t *d = dst.rgba + 4 * ((size_t)y * dw + x);

            for (int c = 0; c < 4; c++) {
                double v = (1 - wx) * (1 - wy) * p00[c]
                         + wx * (1 - wy) * p10[c]
                         + (1 - wx) * wy * p01[c]
                         + wx * wy * p11[c];
                d[c] = clamp8((int)lround(v));
            }
        }
    }
    return dst;
}

/* --- X11 helpers --- */

static int comp_active(Display *d, int s)
{
    char n[64];
    snprintf(n, 64, "_NET_WM_CM_S%d", s);
    Atom a = XInternAtom(d, n, False);
    return a != None && XGetSelectionOwner(d, a) != None;
}

static Visual *find_argb(Display *d, int s, int *dep)
{
    XVisualInfo t;
    t.screen = s;
    t.depth = 32;
    t.class = TrueColor;
    int c = 0;
    XVisualInfo *l = XGetVisualInfo(d,
        VisualScreenMask | VisualDepthMask | VisualClassMask, &t, &c);
    if (!l || !c) {
        if (l) XFree(l);
        return NULL;
    }
    Visual *r = NULL;
    for (int i = 0; i < c; i++) {
        XRenderPictFormat *f = XRenderFindVisualFormat(d, l[i].visual);
        if (f && f->type == PictTypeDirect && f->direct.alphaMask && f->depth == 32) {
            r = l[i].visual;
            *dep = 32;
            break;
        }
    }
    XFree(l);
    return r;
}

static void shape_passthrough(Display *d, Window w)
{
    char z = 0;
    Pixmap p = XCreateBitmapFromData(d, w, &z, 1, 1);
    if (p) {
        XShapeCombineMask(d, w, ShapeInput, 0, 0, p, ShapeSet);
        XFreePixmap(d, p);
    }
}

static Pixmap mask_alpha(Display *d, Drawable dr, const uint8_t *rgba, int w, int h)
{
    int bo = BitmapBitOrder(d);
    int st = (w + 7) / 8;
    uint8_t *b = (uint8_t *)xcalloc((size_t)st * h, 1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (rgba[4 * ((size_t)y * w + x) + 3] > 0) {
                int bi = y * st + (x >> 3);
                int bt = x & 7;
                b[bi] |= (uint8_t)(1u << (bo == LSBFirst ? bt : 7 - bt));
            }
        }
    }
    Pixmap p = XCreateBitmapFromData(d, dr, (char *)b, (unsigned)w, (unsigned)h);
    free(b);
    return p;
}

static Pixmap mask_dither(Display *d, Drawable dr,
                          const uint8_t *rgba, int w, int h, double op)
{
    int bo = BitmapBitOrder(d);
    int st = (w + 7) / 8;
    uint8_t *b = (uint8_t *)xcalloc((size_t)st * h, 1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double ef = ((double)rgba[4 * ((size_t)y * w + x) + 3] / 255.0) * op;
            if (ef > ((double)BAYER8[y & 7][x & 7] + 0.5) / 64.0) {
                int bi = y * st + (x >> 3);
                int bt = x & 7;
                b[bi] |= (uint8_t)(1u << (bo == LSBFirst ? bt : 7 - bt));
            }
        }
    }
    Pixmap p = XCreateBitmapFromData(d, dr, (char *)b, (unsigned)w, (unsigned)h);
    free(b);
    return p;
}

/* --- overlay context --- */

typedef struct {
    Display *dpy;
    int scr;
    Window root, win;
    Visual *vis;
    int depth;
    Colormap cmap;
    Cursor cur_move;
    int w, h;
    int has_shape, has_render, has_comp, use_argb, use_dither;
    int shape_evbase;
    int dragging, drag_rx, drag_ry, win_ox, win_oy;
    XRenderPictFormat *fmtw;
    Picture picw, pics;
    Pixmap pms;
    GC gcpm, gcw;
    XImage *ximg;
    Image *src;
    double opacity;
    int cur_maxdim;
} Ctx;

static void ctx_getpos(Ctx *c, int *ox, int *oy)
{
    Window ch;
    int x = 0, y = 0;
    XTranslateCoordinates(c->dpy, c->win, c->root, 0, 0, &x, &y, &ch);
    if (ox) *ox = x;
    if (oy) *oy = y;
}

static int point_in_win(Ctx *c, int rx, int ry)
{
    int wx = 0, wy = 0;
    ctx_getpos(c, &wx, &wy);
    return rx >= wx && rx < wx + c->w && ry >= wy && ry < wy + c->h;
}

static void ctx_sethints(Ctx *c)
{
    Atom wt = XInternAtom(c->dpy, "_NET_WM_WINDOW_TYPE", False);
    Atom tu = XInternAtom(c->dpy, "_NET_WM_WINDOW_TYPE_UTILITY", False);
    if (wt && tu)
        XChangeProperty(c->dpy, c->win, wt, XA_ATOM, 32,
                        PropModeReplace, (unsigned char *)&tu, 1);

    Atom ws = XInternAtom(c->dpy, "_NET_WM_STATE", False);
    Atom sa = XInternAtom(c->dpy, "_NET_WM_STATE_ABOVE", False);
    if (ws && sa)
        XChangeProperty(c->dpy, c->win, ws, XA_ATOM, 32,
                        PropModeReplace, (unsigned char *)&sa, 1);

    XClassHint ch;
    ch.res_name = (char *)"ghostlay";
    ch.res_class = (char *)"Ghostlay";
    XSetClassHint(c->dpy, c->win, &ch);
}

static void ctx_init(Ctx *c, int w, int h, int dither)
{
    memset(c, 0, sizeof(*c));
    c->dpy = XOpenDisplay(NULL);
    if (!c->dpy) die("cannot open display");

    XSetErrorHandler(x_error);
    XSetIOErrorHandler(xio_error);

    c->scr = DefaultScreen(c->dpy);
    c->root = RootWindow(c->dpy, c->scr);
    c->w = w;
    c->h = h;
    c->use_dither = dither;

    int sev = 0, ser = 0;
    c->has_shape = XShapeQueryExtension(c->dpy, &sev, &ser);
    c->shape_evbase = sev;

    int rev = 0, rer = 0;
    c->has_render = XRenderQueryExtension(c->dpy, &rev, &rer);
    c->has_comp = comp_active(c->dpy, c->scr);

    Visual *v = NULL;
    int dep = 0;
    if (!dither && c->has_render)
        v = find_argb(c->dpy, c->scr, &dep);
    if (v) {
        c->vis = v;
        c->depth = dep;
    } else {
        c->vis = DefaultVisual(c->dpy, c->scr);
        c->depth = DefaultDepth(c->dpy, c->scr);
    }

    c->use_argb = (!dither && c->has_render && c->depth == 32);
    c->cmap = XCreateColormap(c->dpy, c->root, c->vis, AllocNone);

    int sw = DisplayWidth(c->dpy, c->scr);
    int sh = DisplayHeight(c->dpy, c->scr);
    int px = (sw - w) / 2;
    int py = (sh - h) / 2;
    if (px < 0)
        px = 0;
    if (py < 0)
        py = 0;

    XSetWindowAttributes attr;
    memset(&attr, 0, sizeof(attr));
    attr.colormap = c->cmap;
    attr.border_pixel = 0;
    attr.background_pixel = 0;
    attr.override_redirect = True;
    attr.event_mask = ExposureMask | StructureNotifyMask;

    c->win = XCreateWindow(c->dpy, c->root, px, py,
                           (unsigned)w, (unsigned)h,
                           0, c->depth, InputOutput, c->vis,
                           CWColormap | CWBorderPixel | CWBackPixel
                           | CWOverrideRedirect | CWEventMask, &attr);
    if (!c->win) die("cannot create window");

    XStoreName(c->dpy, c->win, "ghostlay");
    ctx_sethints(c);

    c->cur_move = XCreateFontCursor(c->dpy, XC_fleur);

    if (c->has_shape)
        shape_passthrough(c->dpy, c->win);

    /* Alt+Button1 on root for drag (with and without NumLock) */
    XGrabButton(c->dpy, Button1, Mod1Mask, c->root, True,
                ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
                GrabModeAsync, GrabModeAsync, None, c->cur_move);
    XGrabButton(c->dpy, Button1, Mod1Mask | Mod2Mask, c->root, True,
                ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
                GrabModeAsync, GrabModeAsync, None, c->cur_move);

    /* Alt+=/-, Alt+KP_Add/Sub, Alt+Q for resize and quit */
    KeyCode keys[] = {
        XKeysymToKeycode(c->dpy, XK_plus),
        XKeysymToKeycode(c->dpy, XK_minus),
        XKeysymToKeycode(c->dpy, XK_equal),
        XKeysymToKeycode(c->dpy, XK_KP_Add),
        XKeysymToKeycode(c->dpy, XK_KP_Subtract),
        XKeysymToKeycode(c->dpy, XK_q)
    };
    unsigned mods[] = {Mod1Mask, Mod1Mask | Mod2Mask};
    for (int m = 0; m < 2; m++) {
        for (int k = 0; k < 6; k++) {
            if (keys[k])
                XGrabKey(c->dpy, keys[k], mods[m], c->root,
                         True, GrabModeAsync, GrabModeAsync);
        }
    }

    if (c->use_argb) {
        c->fmtw = XRenderFindVisualFormat(c->dpy, c->vis);
        if (!c->fmtw) die("XRender format failed");
        XRenderPictureAttributes pa;
        memset(&pa, 0, sizeof(pa));
        c->picw = XRenderCreatePicture(c->dpy, c->win, c->fmtw, 0, &pa);
    } else {
        c->gcw = XCreateGC(c->dpy, c->win, 0, NULL);
    }

    if (dither)
        fprintf(stderr, "ghostlay: dither+XShape mode\n");
    else if (!c->has_comp)
        fprintf(stderr, "ghostlay: no compositor, use -d or picom\n");

    XMapRaised(c->dpy, c->win);
    XFlush(c->dpy);
}

static void ctx_destroy(Ctx *c)
{
    if (!c || !c->dpy) return;
    if (c->dragging) {
        XUngrabPointer(c->dpy, CurrentTime);
        c->dragging = 0;
    }
    XUngrabButton(c->dpy, Button1, AnyModifier, c->root);
    XUngrabKey(c->dpy, AnyKey, AnyModifier, c->root);
    if (c->ximg) { XDestroyImage(c->ximg); c->ximg = NULL; }
    if (c->gcw) { XFreeGC(c->dpy, c->gcw); c->gcw = 0; }
    if (c->pics) { XRenderFreePicture(c->dpy, c->pics); c->pics = 0; }
    if (c->pms) { XFreePixmap(c->dpy, c->pms); c->pms = 0; }
    if (c->picw) { XRenderFreePicture(c->dpy, c->picw); c->picw = 0; }
    if (c->gcpm) { XFreeGC(c->dpy, c->gcpm); c->gcpm = 0; }
    if (c->cur_move) { XFreeCursor(c->dpy, c->cur_move); c->cur_move = 0; }
    if (c->win) { XDestroyWindow(c->dpy, c->win); c->win = 0; }
    if (c->cmap) { XFreeColormap(c->dpy, c->cmap); c->cmap = 0; }
    XCloseDisplay(c->dpy);
    c->dpy = NULL;
}

/* --- drawing --- */

static void upload_argb(Ctx *c, const uint8_t *rgba, double op)
{
    if (!c->use_argb) return;
    if (op < OPACITY_MIN) op = OPACITY_MIN;
    if (op > 1) op = 1;

    if (c->pics) { XRenderFreePicture(c->dpy, c->pics); c->pics = 0; }
    if (c->pms) { XFreePixmap(c->dpy, c->pms); c->pms = 0; }

    c->pms = XCreatePixmap(c->dpy, c->win,
                           (unsigned)c->w, (unsigned)c->h, 32);
    if (!c->gcpm)
        c->gcpm = XCreateGC(c->dpy, c->pms, 0, NULL);

    XImage *xi = XCreateImage(c->dpy, c->vis, 32, ZPixmap, 0, NULL,
                              (unsigned)c->w, (unsigned)c->h, 32, 0);
    if (!xi) die("XCreateImage failed");

    size_t ds = (size_t)xi->bytes_per_line * xi->height;
    xi->data = (char *)xmalloc(ds);
    memset(xi->data, 0, ds);

    PxFmt pf = make_pxfmt(xi, c->fmtw);
    int ua = c->has_comp;
    for (int y = 0; y < c->h; y++) {
        for (int x = 0; x < c->w; x++) {
            const uint8_t *p = rgba + 4 * ((size_t)y * c->w + x);
            uint8_t r, g, b, a;
            if (ua) {
                a = clamp8((int)lround((double)p[3] * op));
                r = (uint8_t)(((unsigned)p[0] * a + 127) / 255);
                g = (uint8_t)(((unsigned)p[1] * a + 127) / 255);
                b = (uint8_t)(((unsigned)p[2] * a + 127) / 255);
            } else {
                a = 255;
                r = clamp8((int)lround(p[0] * op));
                g = clamp8((int)lround(p[1] * op));
                b = clamp8((int)lround(p[2] * op));
            }
            putpx(xi->data, xi->bytes_per_line, x, y,
                  packpx(&pf, r, g, b, a), pf.bpp);
        }
    }

    XPutImage(c->dpy, c->pms, c->gcpm, xi,
              0, 0, 0, 0, (unsigned)c->w, (unsigned)c->h);
    XDestroyImage(xi);

    XRenderPictFormat *fs = XRenderFindStandardFormat(c->dpy, PictStandardARGB32);
    if (!fs) fs = c->fmtw;
    XRenderPictureAttributes pa;
    memset(&pa, 0, sizeof(pa));
    c->pics = XRenderCreatePicture(c->dpy, c->pms, fs, 0, &pa);
}

static void draw_argb(Ctx *c)
{
    if (!c->use_argb || !c->pics) return;
    XRenderColor cl = {0, 0, 0, 0};
    XRenderFillRectangle(c->dpy, PictOpClear, c->picw, &cl,
                         0, 0, (unsigned)c->w, (unsigned)c->h);
    XRenderComposite(c->dpy, PictOpSrc, c->pics, None, c->picw,
                     0, 0, 0, 0, 0, 0, (unsigned)c->w, (unsigned)c->h);
}

static void build_ximg(Ctx *c, const uint8_t *rgba, double op)
{
    if (c->use_argb) return;
    if (op < OPACITY_MIN) op = OPACITY_MIN;
    if (op > 1) op = 1;
    if (c->ximg) { XDestroyImage(c->ximg); c->ximg = NULL; }

    XImage *xi = XCreateImage(c->dpy, c->vis, (unsigned)c->depth,
                              ZPixmap, 0, NULL,
                              (unsigned)c->w, (unsigned)c->h, 32, 0);
    if (!xi) die("XCreateImage failed");

    size_t ds = (size_t)xi->bytes_per_line * xi->height;
    xi->data = (char *)xmalloc(ds);
    memset(xi->data, 0, ds);

    PxFmt pf = make_pxfmt(xi, NULL);
    for (int y = 0; y < c->h; y++) {
        for (int x = 0; x < c->w; x++) {
            const uint8_t *p = rgba + 4 * ((size_t)y * c->w + x);
            uint8_t r, g, b;
            if (c->use_dither) {
                r = p[0]; g = p[1]; b = p[2];
            } else {
                r = clamp8((int)lround(p[0] * op));
                g = clamp8((int)lround(p[1] * op));
                b = clamp8((int)lround(p[2] * op));
            }
            putpx(xi->data, xi->bytes_per_line, x, y,
                  packpx(&pf, r, g, b, 255), pf.bpp);
        }
    }
    c->ximg = xi;
}

static void draw_ximg(Ctx *c)
{
    if (c->use_argb || !c->ximg) return;
    XPutImage(c->dpy, c->win, c->gcw, c->ximg,
              0, 0, 0, 0, (unsigned)c->w, (unsigned)c->h);
}

static void apply_shape(Ctx *c, const uint8_t *rgba, double op)
{
    if (!c->has_shape) return;
    Pixmap pm = c->use_dither
        ? mask_dither(c->dpy, c->win, rgba, c->w, c->h, op)
        : mask_alpha(c->dpy, c->win, rgba, c->w, c->h);
    if (!pm) return;
    XShapeCombineMask(c->dpy, c->win, ShapeBounding, 0, 0, pm, ShapeSet);
    XFreePixmap(c->dpy, pm);
}

static void redraw(Ctx *c)
{
    if (c->use_argb) draw_argb(c);
    else draw_ximg(c);
    XRaiseWindow(c->dpy, c->win);
    XFlush(c->dpy);
}

static void rebuild_at_size(Ctx *c, int new_maxdim)
{
    if (new_maxdim < MIN_DIM) new_maxdim = MIN_DIM;
    if (new_maxdim > MAX_SCALE_DIM) new_maxdim = MAX_SCALE_DIM;
    c->cur_maxdim = new_maxdim;

    Image scaled = scale_bilinear(c->src, new_maxdim);

    if (c->ximg) { XDestroyImage(c->ximg); c->ximg = NULL; }
    if (c->pics) { XRenderFreePicture(c->dpy, c->pics); c->pics = 0; }
    if (c->pms) { XFreePixmap(c->dpy, c->pms); c->pms = 0; }

    c->w = scaled.w;
    c->h = scaled.h;
    XResizeWindow(c->dpy, c->win, (unsigned)c->w, (unsigned)c->h);

    if (c->has_shape)
        shape_passthrough(c->dpy, c->win);

    if (c->use_argb) {
        upload_argb(c, scaled.rgba, c->opacity);
        if (!c->has_comp)
            apply_shape(c, scaled.rgba, c->opacity);
    } else {
        build_ximg(c, scaled.rgba, c->opacity);
        apply_shape(c, scaled.rgba, c->opacity);
    }

    redraw(c);
    image_free(&scaled);
    fprintf(stderr, "ghostlay: resized %dx%d\n", c->w, c->h);
}

/* --- drain pipe --- */

static void drain_pipe(void)
{
    if (g_wake_rd < 0) return;
    uint8_t b[64];
    while (read(g_wake_rd, b, 64) > 0)
        ;
}

/* --- event loop --- */

static void run_loop(Ctx *c)
{
    int xfd = ConnectionNumber(c->dpy);
    int tick = 0;

    while (g_running) {
        while (XPending(c->dpy) > 0) {
            XEvent ev;
            XNextEvent(c->dpy, &ev);

            if (c->has_shape && ev.type >= c->shape_evbase
                && ev.type < c->shape_evbase + 2)
                continue;

            switch (ev.type) {
            case Expose:
                if (!ev.xexpose.count) redraw(c);
                break;

            case KeyPress: {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_plus || ks == XK_equal || ks == XK_KP_Add)
                    rebuild_at_size(c, c->cur_maxdim + RESIZE_STEP);
                else if (ks == XK_minus || ks == XK_KP_Subtract)
                    rebuild_at_size(c, c->cur_maxdim - RESIZE_STEP);
                else if (ks == XK_q || ks == XK_Q)
                    g_running = 0;
                break;
            }

            case ButtonPress:
                if (ev.xbutton.button == Button1
                    && point_in_win(c, ev.xbutton.x_root, ev.xbutton.y_root))
                {
                    c->dragging = 1;
                    c->drag_rx = ev.xbutton.x_root;
                    c->drag_ry = ev.xbutton.y_root;
                    ctx_getpos(c, &c->win_ox, &c->win_oy);
                }
                break;

            case MotionNotify:
                if (c->dragging) {
                    XEvent last = ev, tmp;
                    while (XCheckTypedEvent(c->dpy, MotionNotify, &tmp))
                        last = tmp;
                    int dx = last.xmotion.x_root - c->drag_rx;
                    int dy = last.xmotion.y_root - c->drag_ry;
                    XMoveWindow(c->dpy, c->win, c->win_ox + dx, c->win_oy + dy);
                    XRaiseWindow(c->dpy, c->win);
                    XFlush(c->dpy);
                }
                break;

            case ButtonRelease:
                if (ev.xbutton.button == Button1 && c->dragging)
                    c->dragging = 0;
                break;

            default:
                break;
            }
        }

        if (!g_running) break;

        fd_set rf;
        FD_ZERO(&rf);
        FD_SET(xfd, &rf);
        int mx = xfd;
        if (g_wake_rd >= 0) {
            FD_SET(g_wake_rd, &rf);
            if (g_wake_rd > mx) mx = g_wake_rd;
        }

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = SELECT_US;

        int r = select(mx + 1, &rf, NULL, NULL, &tv);
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (g_wake_rd >= 0 && FD_ISSET(g_wake_rd, &rf)) {
            drain_pipe();
            break;
        }
        if (!g_running) break;

        tick++;
        if (tick >= RAISE_TICKS) {
            tick = 0;
            XRaiseWindow(c->dpy, c->win);
            XFlush(c->dpy);
        }
    }
}

/* --- usage --- */

static void usage(FILE *o)
{
    fprintf(o,
        "ghostlay v%s — click-through image overlay for X11\n\n"
        "Usage: ghostlay [-o 0..10] [-s PX] [-d] <image>\n\n"
        "  -o N   Opacity 0..10 (default 10)\n"
        "  -s PX  Max dimension in pixels\n"
        "  -d     Dither transparency (no compositor needed)\n"
        "  -h     Help\n\n"
        "Controls:\n"
        "  Alt + Drag        Move overlay\n"
        "  Alt + Plus/Minus  Resize overlay\n"
        "  Alt + Q           Quit (works even in background)\n"
        "  Ctrl+C / SIGTERM  Quit (foreground only)\n\n"
        "Background exit:\n"
        "  Alt+Q  or  killall ghostlay\n\n"
        "Examples:\n"
        "  ghostlay -o 5 -s 500 ref.png\n"
        "  ghostlay -d -o 4 -s 400 ref.png\n"
        "  ghostlay -o 6 -s 800 img.jpg &\n",
        VERSION);
}

/* --- pid file --- */

static char g_pidpath[512] = {0};

static void write_pid(void)
{
    const char *rt = getenv("XDG_RUNTIME_DIR");
    if (!rt) rt = "/tmp";
    snprintf(g_pidpath, 512, "%s/ghostlay.pid", rt);
    FILE *f = fopen(g_pidpath, "w");
    if (f) {
        fprintf(f, "%d\n", (int)getpid());
        fclose(f);
    }
}

static void rm_pid(void)
{
    if (g_pidpath[0]) unlink(g_pidpath);
}

/* --- main --- */

int main(int argc, char **argv)
{
    int olev = 10, tsz = 0, dith = 0, opt;

    while ((opt = getopt(argc, argv, "o:s:dh")) != -1) {
        switch (opt) {
        case 'o': {
            char *e;
            long v = strtol(optarg, &e, 10);
            if (!e || *e) {
                fprintf(stderr, "ghostlay: bad -o\n");
                return 2;
            }
            if (v < 0) v = 0;
            if (v > 10) v = 10;
            olev = (int)v;
            break;
        }
        case 's': {
            char *e;
            long v = strtol(optarg, &e, 10);
            if (!e || *e || v <= 0) {
                fprintf(stderr, "ghostlay: bad -s\n");
                return 2;
            }
            if (v > MAX_SCALE_DIM) v = MAX_SCALE_DIM;
            tsz = (int)v;
            break;
        }
        case 'd':
            dith = 1;
            break;
        case 'h':
            usage(stdout);
            return 0;
        default:
            usage(stderr);
            return 2;
        }
    }

    if (optind >= argc) {
        usage(stderr);
        return 2;
    }
    const char *path = argv[optind];
    if (access(path, R_OK)) {
        fprintf(stderr, "ghostlay: %s: %s\n", path, strerror(errno));
        return 1;
    }

    /* Auto-exit when parent shell dies (Linux-specific, safe no-op elsewhere) */
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() == 1) {
        fprintf(stderr, "ghostlay: parent already gone\n");
        return 0;
    }

    write_pid();

    int pfd[2];
    if (!pipe(pfd)) {
        g_wake_rd = pfd[0];
        g_wake_wr = pfd[1];
        fcntl(g_wake_rd, F_SETFL, O_NONBLOCK);
        fcntl(g_wake_wr, F_SETFL, O_NONBLOCK);
        fcntl(g_wake_rd, F_SETFD, FD_CLOEXEC);
        fcntl(g_wake_wr, F_SETFD, FD_CLOEXEC);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    Image src = load_image(path);
    Image scaled = scale_bilinear(&src, tsz);

    double op = olev == 0 ? OPACITY_MIN : (double)olev / 10.0;

    /*
     * cur_maxdim tracks the current longer side of the displayed image.
     * This is the value that Alt+/- adjusts by RESIZE_STEP.
     */
    int cur_maxdim = scaled.w > scaled.h ? scaled.w : scaled.h;

    fprintf(stderr, "ghostlay: %dx%d -> %dx%d, opacity %d/10%s (pid %d)\n",
            src.w, src.h, scaled.w, scaled.h, olev,
            dith ? ", dither" : "", (int)getpid());
    fprintf(stderr, "ghostlay: Alt+Drag=move  Alt+/-=resize  Alt+Q=quit\n");

    Ctx ctx;
    ctx_init(&ctx, scaled.w, scaled.h, dith);
    ctx.src = &src;
    ctx.opacity = op;
    ctx.cur_maxdim = cur_maxdim;

    if (ctx.use_argb) {
        upload_argb(&ctx, scaled.rgba, op);
        if (!ctx.has_comp)
            apply_shape(&ctx, scaled.rgba, op);
    } else {
        build_ximg(&ctx, scaled.rgba, op);
        apply_shape(&ctx, scaled.rgba, op);
    }

    redraw(&ctx);
    run_loop(&ctx);

    ctx_destroy(&ctx);
    image_free(&src);
    image_free(&scaled);
    if (g_wake_rd >= 0) close(g_wake_rd);
    if (g_wake_wr >= 0) close(g_wake_wr);
    rm_pid();
    fprintf(stderr, "ghostlay: exit\n");
    return 0;
}
