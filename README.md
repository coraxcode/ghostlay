Compilation:

gcc -O2 -std=c11 -Wall -Wextra -Wpedantic ghostlay.c -o ghostlay -lX11 -lXext -lXrender -lpng -ljpeg -lm

  Usage:
  
  ./ghostlay -o 6 -s 500 ./name_of_image.jpg & picom
