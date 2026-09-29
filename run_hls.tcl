# my_window — HLS ap_fixed<8,4> window clustering (Vitis HLS 2023.2)
set proj_name my_window_proj
set top_name  my_window
set part_name xcu250-figd2104-2L-e
set clk_ns    10

open_project -reset $proj_name
set_top $top_name

add_files my_window.h
add_files my_window.cpp

add_files -tb my_window_tb.cpp
add_files -tb my_window.h
add_files -tb data

open_solution -reset solution1 -flow_target vivado
set_part $part_name
create_clock -period $clk_ns -name default

csim_design
csynth_design
exit
