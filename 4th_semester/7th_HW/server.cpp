#include <iostream>
#include <thread>

#include <boost/asio.hpp>


class Server {

public:

    Server(std::size_t max_queue_size, int port) {

        receiveNickname();

        boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address_v4::any(), port);

        try {

            boost::asio::ip::tcp::acceptor acceptor(io_service, endpoint.protocol());

            acceptor.bind(endpoint);

            acceptor.listen(max_queue_size);

            acceptor.accept(socket);

            std::cout << "Connection set, you can start chatting!" << std::endl << std::endl;

            end = false;

            thread = std::thread(&Server::writeMessages, this);

            readMessages();

            thread.join();

            system("pause");
        }
        catch (boost::system::system_error & e) {
            std::cout << "Error occured! Error code = " << e.code() << ". Message: " << e.what() << std::endl;

            system("pause");


        }
    }

    ~Server() = default;

private:

    boost::asio::io_service io_service;
    std::string nickname;
    boost::asio::ip::tcp::socket socket = boost::asio::ip::tcp::socket(io_service);
    boost::asio::streambuf buffer;
    std::thread thread;
    bool end;


    void receiveNickname() {
        std::cout << "Enter your nickname:\n";
        std::cin >> nickname;
    }


    void writeMessages () {

        std::string buf;
        std::string message;

        while (true) {

            std::getline(std::cin, buf);

            if (end) break;

            if (buf.empty()) continue;

            if (buf == "exit") {
                message = buf + '\n';

                boost::asio::write(socket, boost::asio::buffer(message));

                break;
            }

            message = "[" + nickname + "]: " + buf + '\n';

            boost::asio::write(socket, boost::asio::buffer(message));

        }

        end = true;

    }

    void readMessages () {

        // Printing new messages
        while (true) {

            boost::asio::read_until(socket, buffer, '\n');

            std::string message;

            std::istream input_stream(&buffer);
            std::getline(input_stream, message, '\n');

            if (message == "exit") {
                if (!end) {

                    end = true;

                    message = "exit\n";
                    boost::asio::write(socket, boost::asio::buffer(message));

                    std::cout << std::endl << "Your interlocutor close his program, press enter to continue...\n";
                }

                break;
            }


            std::cout << message << "\n";

        }

    }

};


int main() {

    system("chcp 1251");


    const std::size_t size = 30;

    int port = 3333;

    Server server(size, port);


    return EXIT_SUCCESS;
}
