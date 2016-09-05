a=1
for i in $1*.$2; do
  new=$(printf "$3%01d.png" "$a") #04 pad to length of 4
  mv -- "$i" "$new"
  let a=a+1
done

