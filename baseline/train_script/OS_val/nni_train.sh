while [[ ! -z $CHECK ]]; do
    PORT=$(( ( RANDOM % 60000 )  + 1025 ))
    CHECK=$(sudo netstat -ap | grep $PORT)
done

echo $PORT

python3 ./baseline/train_val_OS.py 