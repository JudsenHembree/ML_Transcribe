run: clean
	python src/app.py --new --record
clean:
	rm -rf data/*
