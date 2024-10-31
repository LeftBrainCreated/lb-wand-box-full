sleep 2

xdotool key Ctrl+Alt+t

echo "Waiting..."

sleep 3

#xdotool type "bash run-main.sh"
#xdotool type "source Documents/.venv/bin/activate"
#xdotool key Return

#xdotool type "python Documents/src/lb_wand_box/SVM_Model/run_main.py"

code /home/leftbrain/Documents/src/lb_wand_box/SVM_Model/run_main.py

echo "opening file"
sleep 10

WINDOW_ID = (xdotool search --name "Visual")

xdotool windowactivate $WINDOW_ID  
sleep 2

xdotool key F5
