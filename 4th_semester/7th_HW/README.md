# Answers on questions: #
## How is information transferred between devices on the Internet? ##

There are two main concepts used on the Internet: address and protocol. Every computer connected to the Internet has its own unique address. Even with a temporary connection, a unique address is allocated to the computer. At any given time, all computers connected to the Internet have different addresses. Just as a mailing address uniquely identifies a person's location, an Internet address uniquely identifies a computer's location on a network.

Protocols are standards that define the forms of presentation and methods of sending messages, the procedures for their interpretation, the rules for the joint operation of various equipment in networks.

It is almost impossible to describe all the rules of interaction in one protocol. Therefore, network protocols are built on a multi-level principle. For example, the lower level describes the rules for transferring small pieces of information from one computer to another, since it is much easier to track the correctness of the transfer of small pieces of information. If some part of the information was distorted by interference during transmission, then at this level only the distorted part is requested to be retransmitted. The next level protocol describes how to break large data sets into small pieces and put them back together. In this case, small parts are sent using the lower layer protocol. The next higher level describes file transfer. In this case, the protocols of the lower layers are used. Thus, to implement a new high-level protocol on the Internet, one does not need to know the features of the network, but one must be able to use lower-level protocols.

The analogy of layered protocols can be found in everyday life. For example, you can send the text of a document while talking on the phone. However, you do not need to know how the telephone network works. You know you just need to dial a number and wait for the other person to pick up the phone.
You can use fax to send an image of a document. You insert a document into the fax machine, dial the phone number of another fax machine, and send the document. In this case, you may not even think about how the image of the document is transmitted over telephone lines. You're just using a high-level protocol: "insert document into fax machine, dial number, press start button on fax machine". In doing so, you have used at least two more protocol layers: the telephone network operation protocol and the fax transmission protocol.

Similarly, the Internet has several layers of protocols that interact with each other. At the lower level, two main protocols are used: IP - Internet Protocol (Internet Protocol) and TCP - Transmission Control Protocol (Transmission Control Protocol). Since these two protocols are closely related, they are often combined, and the Internet is said to have TCP/IP as the underlying protocol. All other numerous protocols are built on the basis of the TCP / IP protocols.

### TCP protocol ###

The TCP protocol breaks information into parts (packets) and enumerates all these parts so that when received, information can be correctly collected. Also, when disassembling a wooden frame, the logs are numbered in order to quickly assemble the house in another place. Then, using the IP protocol, all parts are transmitted to the recipient, where it is checked using the TCP protocol whether all parts have been received. Since individual parts can travel across the Internet in a variety of ways, the order of arrival of the parts can be broken. After receiving all the parts, TCP arranges them in the right order and assembles them into a single whole.

### IP protocol ###

For the TCP protocol, it doesn't matter which way information travels over the Internet. This is what the IP protocol does. In the same way as when transporting individual numbered logs, it does not matter which way they are being transported. To each piece of information received, the IP protocol adds service information, from which you can find out the addresses of the sender and recipient of information. If we follow the analogy with mail, then the data is placed in an envelope or package on which the recipient's address is written. Further, the IP protocol, just like regular mail, ensures the delivery of all packets to the recipient. At the same time, the speed and paths of passage of different envelopes can be different. The Internet is often depicted as a blurry cloud. You do not know the path of information, but properly formatted IP - packets reach the recipient.

## Why doesn't the transfer of large amounts of data (for example, an archive with a machine learning dataset) over the Internet delay the exchange of data between other devices on the network? ##

It seems that this is simply because on the Internet there are nodes for transferring information between users and the transfer of information from user to user occurs through such "intermediaries" (there may be several). When information is transmitted from subscriber 1 to subscriber 2, other network participants do not participate in this. But with a large number of requests, the server can still be overloaded, and then all network participants connected to this server will feel it. Also, big data is split into smaller pieces before being sent, so there is no continuous transfer of large files and it also saves third-party users from delays.

# Description of the written code #

This folder contains two C++ files - Client and Server. This implementation needs the Boost libraries to be installed. With the help of boost-asio, a local text messaging system is created based on processes and network interaction mechanisms, consisting of two processes, a client and a server, that support data exchange via sockets using the TCP protocol. An example of work is presented in the file example.jpg.

* The client.cpp file contains the client implementation for creating a chat with the server. To run use:
> g++ client.cpp -pthread -o client; ./client

* The server.cpp file contains the implementation of the server for creating a chat. To run use: 
> g++ server.cpp -pthread -o server; ./server
