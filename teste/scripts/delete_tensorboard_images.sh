#!/bin/bash

directory=$1;

if [[ -z $1 ]];
then
    directory='logs/Images';
    rm -rf "$directory";
else
    rm -rfi "$directory";

fi

echo "Directory: $directory removed successfully!";

exit;