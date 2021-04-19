******** Basic Setup for the Online Hex Tournament (OHT) **************

What follows are basic instructions for preparing to play the OHT.  You will run 
a client object on your machine that will interact with our server object (on another 
machine).  We are supplying you with all of the important interface code, so you should
 only need to think about coming up with smart HEX moves.  
 
 Follow the steps below to get setup,
 then follow the more detailed instructions in the main project specification; many of those
 instructions are repeated as comments in the file BasicClientActor.py (see below).
 
 1)  Go to the course web page under the "Materials" section - the same place where
 you found this TXT file.  There you will find a
     ZIP file containing all of the code that we supply you.  The main zipped files are:
     
 		a) server.crt
 		b) BasicClientActorAbs.py
 		c) BasicClientActor.py
 
 2) Put all 3 files in the directory containing all of your other Hex-player files.

 3) Modify BasicClientActor.py so that your version can respond to our server's request
 for your next move.  As explained in the main project specification, you MUST modify one of
  the methods in that file.  All of the other methods are sufficient as they are, 
  although you might want to expand them as well, depending upon how much information 
  about the game that you want to save and analyze.
  
  See the project description for more details about the required and optional 
  modifications.
 
 4)  Do NOT modify the other 2 files, especially server.crt, which is a certificate that
 gives you permission to play against our server. 
 
 5) You need to be logged onto an NTNU server, either directly or via VPN.  We use
 NTNU's system for recognizing your username and password, but we do not read or save
 any password information in the OHT. 
 
 6) To start an interaction with the server, simply do the following 2 operations:
 
        a) Create a BasicClientActor object.
        b) Call the method "connect_to_server", which your BasicClientActor object will
            inherit from BasicClientActorAbs.
            
 7) The two steps above are already done for you at the bottom of the BasicClientActor.py
 file, so if you just start that file when you start python, your client will immediately
 begin interacting with the server.  So just type:
 
 	python3 BasicClientActor.py
 	
 	on the command line, and the client-server connection will begin.
 	
 8) Further details of the client-server interaction are given in the main 
 project specification.
 
