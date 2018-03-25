clear
reset
unset key
unset tics
unset border
set datafile separator ','
set terminal pngcairo size 8000, 6000 lw 2.0  enhanced font "Arial-Bold, 70"
set output 'bldg.png'
set xyplane at 0.0
set view equal xyz

set xlabel '' offset 0, 0, 0
set ylabel '' offset 0, 0 , 0
set zlabel '' offset 0, 0 ,0
#set zlabel 'Height (Feet)' offset 0, 0, 0 rotate by 90

set xrange [0: 12000.0]
#set xtics 0, 500, 1500 offset 0, -0.5, 0
set format x ""
set format y ""
set format z ""
#set grid

set yrange [0: 10000.0]
#set ytics 0, 500, 1500

set zrange [-100: 5000.0]
#set ztics 0, 1000, 4000

set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0


set view 70, 15, 1.0, 2.0

splot \
        'world_trade_center.losnu'  using 1:2:(4500):($4-$1):($5-$2):($6-$3) w vectors back nohead  lc rgbcolor 'red' lw 0.1 ,\
        'world_trade_center.bldgnu' using 1:2:3 w lines lc rgbcolor 'black' lw 0.1



######################################small bldgs ###########################################################################


clear
reset
unset key
#unset tics
#unset border

set datafile separator ','
set terminal pngcairo dashed size 1500, 2000 lw 3.0  enhanced font "Arial-Bold, 35"
set output 'bldg_small.png'
set xyplane at 0.0
set view equal xyz

set xlabel 'Feet' offset 0, -0.25, 0
set ylabel 'Feet'  offset -0.25, 0 , 0
set ylabel rotate by 90
set zlabel 'Height (Feet)' offset 0, 0, 0 rotate by 90

set xrange [500:  1500.0]
set xtics 1000, 500, 1500 offset 0, -0.5, 0

set yrange [1800: 2500.0]
set ytics 2000, 500, 2500  offset 0.5, 0, 0 rotate by 90

set zrange [-100: 2000.0]
set ztics 0, 1000, 3000 rotate by 45

set lmargin 0
set rmargin 0
set tmargin 0
set bmargin 0


set view 55, 25, 1.0, 1.0


splot \
        'world_trade_center.losnu'  using 1:2:($3):($4-$1):($5-$2):($6-$3) w vectors back nohead  lc rgbcolor 'red' lw 2 lt 1,\
        'world_trade_center.bldgnu' using 1:2:3 w lines lc rgbcolor 'black' lw 1.0 lt 1



#################################################################################################################
clear
reset
unset key

set datafile separator ','
set terminal pngcairo truecolor  size 4000, 4000 lw 2.5  enhanced font "Arial-Bold, 80"
set output 'coverage.png'

set xrange [0:12000]
set xtics 0, 2000, 12000
set yrange [0:10000]
set ytics 0, 2000, 12000

plot 'world_trade_center.covnu' using 1:2:($1*0+328.084) with circles lw 2.0 lc rgb "blue" fs transparent solid 0.18 noborder ,\
      'world_trade_center.roofnu' using 1:2 w lines lc rgbcolor 'black' lw 2.0




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

