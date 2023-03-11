run: clean
	python src/app.py --new --graph
clean:
	rm -rf data/*
