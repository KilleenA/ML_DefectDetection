#GNUplot file for plotting cell orientation data and detected defects, GNUplot
#software can be downloaded from: https://sourceforge.net/projects/gnuplot/

#Define file numbers to be plotted
start = 0
stop = 100

#Choose between png or svg image (uncomment desired file type)
set terminal pngcairo rounded size 800,800
#set terminal svg rounded size 800,800

#Create line templates
set style line 1 lt 3 lc rgb 'black' pt 7 lw 3
set style line 2 lt 2 lc rgb 'green' pt 7 lw 2 ps 2
set style line 3 lt 2 lc rgb 'blue' pt 7 lw 2 ps 2
set style arrow 1 head filled size screen 0.01,30,45 ls 4

#Set figure range and unset tic markers
#Use set up from numerical simulation to to determine x and y limits
N=400
d = sqrt(2/sqrt(3))
h = sqrt(3)*d/2
set xrange [0:d*sqrt(N)]
set yrange [0:h*sqrt(N)]
unset xtics
unset ytics
set key right top box opaque

#Filepaths to appropriate files, and set appropriate separator
set datafile separator ","
cell_files(n) = sprintf('./CellFiles/points%06d.txt', n)
posdefect_files(n) = sprintf('./DefectFiles/PosDefects/posdefects%06d.txt', n)
negdefect_files(n) = sprintf('./DefectFiles/NegDefects/negdefects%06d.txt', n)

#Plot data, also plotting the orientation of each cell's long axis
do for [i = start:stop] {
    outfile = sprintf('./Figures/figure%06d.png',i) #Location file will be wrttien to
    set output outfile
    #unset key
    plot cell_files(i) using 1:2:(0.35*cos($3)):(0.35*sin($3)) with vectors nohead ls 1 notitle, \
         cell_files(i) using 1:2:(-0.35*cos($3)):(-0.35*sin($3)) with vectors nohead ls 1 notitle, \
         posdefect_files(i) using 2:1 ls 3 title "+1/2 Defects", \
         negdefect_files(i) using 2:1 ls 2 title "-1/2 Defects"
}
