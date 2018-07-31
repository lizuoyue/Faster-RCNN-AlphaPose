set_time=$2
while true; do
	time='date +%Y%m%d%H%M%S'
	if [ '$time' -ge '$set_time' ]; then
		break
	fi
echo 'dashabi'