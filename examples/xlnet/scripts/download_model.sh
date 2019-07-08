#!/usr/bin/env bash
wget -P pretrained/ https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
cd pretrained/
unzip cased_L-24_H-1024_A-16.zip
rm cased_L-24_H-1024_A-16.zip
