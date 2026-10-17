#!/bin/bash
set -e

if [ "${DOWNLOAD}" == "true" ]; then
  echo "Downloading the data"
  mkdir -p input
  cd input
  wget https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

  cd ../splits
  wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
  cd ..

fi
  cd input
  mkdir categories
  cd categories
  unrar x ../hmdb51_org.rar
  for i in *.rar; do
    unrar x $i
  done
  cd ../..
  cd splits
  unrar x test_train_splits.rar
  cd ..
