#!/bin/bash  
nohup python rename_resize_convert_fast.py -p $1 -o $2 > output.log 2>&1 &  
