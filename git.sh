#!/usr/bin/expect
set action [lindex $argv 0]
spawn git $action origin master
expect "Username"
send "917563379@qq.com\n"
expect "Password"
send "\n"
expect eof

