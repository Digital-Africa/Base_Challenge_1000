
path="/Users/py3/data science/digital_africa/Base_Challenge_1000/"

find *.py "$path" | while read line; do
	grep 'import' "$line"
done