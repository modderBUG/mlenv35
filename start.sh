#!/bin/bash
# 安装conda环境脚本
echo ok
echo ok
function getSomething(){
  echo 'cd ok'
}
getSomething
function setEnv() {
  echo "当前环境有："
  conda-env list
  echo "创建环境："
  conda create -n tf20 python=3.7
  pip install requests
  echo "ok! press any key to exit! you will exec:"
  echo "conda activate tf20"
  read a
  #pip install requirments.txt
}
