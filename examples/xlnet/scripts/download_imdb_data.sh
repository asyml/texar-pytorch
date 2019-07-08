#!/usr/bin/env bash
DATA_DIR="${DATA_DIR:-data/}"
mkdir -p $DATA_DIR
cd $DATA_DIR
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxf aclImdb_v1.tar.gz
mv aclImdb IMDB
rm aclImdb_v1.tar.gz
