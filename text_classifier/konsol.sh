
path="/Users/py3/data science/digital_africa/Base_Challenge_1000/"

find *.py "$path" | while read line; do
	grep 'import' "$line"
done

echo "alias pip=/usr/local/bin/pip3" >> ~/.zshrc  
echo "alias pip=/usr/local/bin/pip3" >> ~/.bashrc