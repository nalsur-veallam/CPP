#Answers on questions:
##What are data exchange formats like XML and JSON good for?

JSON and XML serve two different purposes.

JSON is a data interchange format, its purpose is to facilitate the exchange of structured data. This is achieved by directly representing the objects, arrays, numbers, strings, and booleans that are often present in the source environment and the destination.

XML, on the other hand, is a markup language, its purpose is document markup. While XML is easy to learn for data interchange, there is a lot more to learn if you want to learn how to use XML as a markup language. For example, you need to learn how to create a document type definition, how to use XSLT to transform XML documents into another form, and how to query XML documents using XPath.

##Tell about the capabilities of the filesystem library. 

The Filesystem library provides facilities for performing operations on file systems and their components, such as paths, regular files, and directories and etc.

     *file: a file system object that holds data, can be written to, read from, or both. Files have names, attributes, one of which is file type: 
        *directory: a file that acts as a container of directory entries, which identify other files (some of which may be other, nested directories). When discussing a particular file, the directory in which it appears as an entry is its parent directory. The parent directory can be represented by the relative pathname "..".
        *regular file: a directory entry that associates a name with an existing file (i.e. a hard link). If multiple hard links are supported, the file is removed after the last hard link to it is removed.
       *path: sequence of elements that identifies a file. It begins with an optional root-name (e.g. "C:" or "//server" on Windows), followed by an optional root-directory (e.g. "/" on Unix), followed by a sequence of zero or more file names (all but last of which have to be directories or links to directories). The native format (e.g. which characters are used as separators) and character encoding of the string representation of a path (the pathname) is implementation-defined, this library provides portable representation of paths. 
       
#Description of the written code

*The date_browser.cpp file contains code that reads the date.txt file and using regular functions checks the data written there against the date pattern DD.MM.YYYY

*The class.cpp file contains code with a structure that stores the name, age, sex, email, city, height, and date of birth of a person (it has its own overloaded operator). Also in the savedata/ directory are stored two json's files with the data of two people recorded there.
