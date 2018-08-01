set_time=$1
while true; do
	time="`date +%Y%m%d%H%M%S`"
	if (("$time" >= "$set_time"))
	then
		break
	fi
done
CUDA_VISIBLE_DEVICES=3 python demo-alpha-pose.py --outputpath=prediction --inputlist=img_list.txt
