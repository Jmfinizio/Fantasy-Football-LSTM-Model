.PHONY: install run clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run the Flask app
run:
	FLASK_APP=app.py flask run --host=0.0.0.0 --port=5050

# Remove temporary and cache files
clean:
	rm -rf __pycache__ *.pyc static/*.png
 