# Useful commands

Needed to authenticate google client
`gcloud auth application-default login`

Format disk
`sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/DEVICE_ID`

Mount disk
`sudo mount -o discard,defaults /dev/sdb /folder/directory`

Check memory usage
`df -h`

Check where disk is mounted
`lsblk` 

Start jup
`jupyter notebook --allow-root --ip=0.0.0.0 --port=8888 --no-browser &`

IP
http://35.185.10.67:8888/notebooks/


sudo mount -o discard,defaults /dev/sdb /home/project_x/data