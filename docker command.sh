# jupyter
docker run --rm -d -p 9000:8888 -v /home/hchuang/Workspaces:/home/jovyan -e JUPYTER_ENABLE_LAB=yes -e GRANT_SUDO=yes --user root jupyter/datascience-notebook start-notebook.sh --NotebookApp.token='cjlab'

# bert
docker build -t huang/crs_bert .
docker run --runtime=nvidia -v /home/huang/Workspaces/crs:/crs -u $(id -u):$(id -g) --name crs_bert -it huang/crs_bert
docker start crs_bert
docker exec -it crs_bert bash

# elmo
docker run --name elmo --runtime=nvidia --rm -it -v /home/hchuang/Workspaces/crs/:/crs_elmo huang/elmo
python setup.py install