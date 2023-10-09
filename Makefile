run-mongo:
    docker run -v $(pwd)/mongodb:/data/db -v $(pwd)/test.bson:/test.bson -d -p 27017:27017 --name m1 mongo