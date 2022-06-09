scp $1/tmp/imago .
convert -size 1280x720 -depth 8 rgba:imago imago.png
