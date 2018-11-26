#!/bin/bash

cd ../data/

echo "Downloading Kaggle Data"
kaggle datasets download -d snapcrack/all-the-news

echo "Extracting Articles ZIP File"
unzip all-the-news.zip