for i in $(cat .gitignore | grep -iv pdf); do rm -f $i; done
rm -f gem_expo_sw.pdf
