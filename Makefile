image_name = ml_transcribe:latest
username = feastertas
pwd = $(shell pwd)

build: #clean_submissions
	docker build . -t ${username}/$(image_name)

push: build
	docker login -u "feastertas" -p "2311IsEZ!" docker.io
	docker push ${username}/${image_name}

run:
	docker run --rm -it --device /dev/snd -v ${pwd}/data:/data ${username}/${image_name} /bin/bash
