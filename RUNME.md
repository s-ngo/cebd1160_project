### To build:

The container can be built using the following command from within this directory:

```
docker build -t s-ngo/cebd1160_project .
```

### To run:

The software can be run from within this directory with the following command:

```
docker run -ti -v ${PWD}/data:/data -v ${PWD}/figures:/figures s-ngo/cebd1160_project /data /figures
```
