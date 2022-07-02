#!/bin/bash -xue

./build.sh

VERSION=$(git describe --always HEAD)
NAME=zm-aidect-$VERSION-debian11-x86_64

mv artifact $NAME
TAR=$NAME.tar.gz
tar czvf $TAR $NAME
gpg --detach-sign $TAR
