Install packages:

sudo apt update
sudo apt install libxrender-dev libx11-dev
sudo apt install libpng-dev
sudo apt install libjpeg-dev

Compilation:

gcc -O2 -std=c11 -Wall -Wextra -Wpedantic ghostlay.c -o ghostlay -lX11 -lXext -lXrender -lpng -ljpeg -lm

  Usage:
  
  ./ghostlay -o 6 -s 500 ./name_of_image.jpg & picom
