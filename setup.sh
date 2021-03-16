#!bin/bash

dir="$PWD/dataset"   # (ref) https://unix.stackexchange.com/questions/188182/how-can-i-get-the-current-working-directory
echo $dir

if [ ! -d $dir ]; then
    mkdir $dir
fi

# (ref) https://stackoverflow.com/questions/12716976/wget-and-changing-directory-in-a-bash-script
if [ ! -e "$dir/ava_v2.2.zip" ]; then
    (cd "$dir" && wget "https://research.google.com/ava/download/ava_v2.2.zip")  
fi

(cd "$dir" && unzip "./ava_v2.2.zip" -d "$dir/ava_v2.2")
