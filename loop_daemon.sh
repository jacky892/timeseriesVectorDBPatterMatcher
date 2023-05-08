py=$1
if [ "$py" = "" ]; then
	echo "$0  <py script>"
	exit
fi
while [ 1 -eq 1 ]
do
#	python tgChatDaemon.py
lc=` ps -ef  | grep $py |  grep -v grep | grep -v vi | grep -v $0 | wc -l`
echo "lc is $lc"
if [ $lc -lt 1 ]; then
	    echo "not running, proceed to run $py"
		python $py
		sleep 20
		echo "started $py"
    else
	        echo 'already running sleep 20 secs'
fi
	sleep 20
done
