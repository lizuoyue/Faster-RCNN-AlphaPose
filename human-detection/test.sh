set_time=$2
while true; do
	time="`date +%Y%m%d%H%M%S`"
	if (("$time" <= "$set_time"))
	then
		break
	fi
done
echo "dashabi"
