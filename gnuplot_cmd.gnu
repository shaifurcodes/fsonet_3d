clear
reset
unset key
set datafile separator ','
set terminal png size 4000, 4000 lw 3.0  enhanced font "Arial-Bold, 70"
set output 'bldg_2.png'
set xyplane at 0.0
set view equal xyz

set xlabel '' offset 0, 0, 0
set ylabel '' offset 0, 0 , 0
set zlabel 'Height (Feet)' offset 0, 0, 0 rotate by 90

set xrange [300:2100.0]
#set xtics 0, 500, 1500 offset 0, -0.5, 0
set format x ""
set format y ""
#set format z ""
set grid

set yrange [500: 2500.0]
#set ytics 0, 500, 1500

set zrange [-100: 2500.0]
set ztics 0, 500, 2500

set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0


#set view 70, 25, 1.0, 2.0

splot \
        'world_trade_center.losnu'  using 1:2:($3):($4-$1):($5-$2):($6-$3) w vectors back nohead  lc rgbcolor 'red' lw 2.0 lt 0,\
        'world_trade_center.bldgnu' using 1:2:3 w lines lc rgbcolor 'black' lw 3.0



#################################################################################################################


splot 'circle.dat' with circles lc rgb "blue" fs transparent solid 0.15 noborder

circle_radius=100
circle_x=0
circle_y=0
circle_z=0
splot "+" using (circle_x+circle_radius*cos(2*pi*$0/99)):(circle_y+circle_radius*sin(2*pi*$0/99)):(circle_z) w l  lc rgb "red" fs transparent solid 0.15 noborder

set datafile separator ','
splot "circle.dat" linecolor "blue" pt 6 ps 100

set style fill transparent solid 0.2 noborder
plot 'circle.dat' using 1:2:(sqrt($3)) with circles, \
     'circle.dat' using 1:2 with linespoints

