class subvector{
private:
    int *mas;
    unsigned int top;
    unsigned int capacity;
};
/*bool init(subvector *qv); //инициализация пустого недовектора


bool push_back(subvector *qv, int d); //добавление элемента в конец недовектора
                                      //с выделением дополнительной памяти при необходимости

int pop_back(subvector *qv); //удаление элемента с конца недовектора


bool resize(subvector *qv, unsigned int new_capacity); //увеличить емкость недовектора


void shrink_to_fit(subvector *qv); //очистить неиспользуемую память


void clear(subvector *qv); //очистить содержимое недовектора, занимаемое место
                           //при этом не меняется

void destructor(subvector *qv); //очистить всю используемую память, инициализировать
                                //недовектор как пустой

bool init_from_file(subvector *qv, char *filename); //инициализировать недовектор из файла
*/
public:
    bool init(){
        top = 0;
        capacity = 0;
        mas = NULL;
        return true;
    }
    
    bool push_back(int d){
        if(top == capacity){
            if(top == 0){
                resize(1);
            }
            else{
                resize(1 + capacity);
            }
        }
        *(mas + top) = d;
        ++top;
        return true;
    }
    
    int pop_back(){
        if(top == 0){
            return 0;
        }
        --top;
        return *(mas + top);
    }


    bool resize(unsigned int new_capacity){
        if(new_capacity < top) {
            return false;
        }
        int *new_mas = new int[new_capacity];
        for (int i = 0; i < top; ++i) {
            *(new_mas + i) = *(mas + i);
        }
        if(mas){
            delete[] mas;
        }
        mas = new_mas;
        capacity = new_capacity;
        return true;
    }


    void shrink_to_fit(){
        resize(top);
    }


    void clear(){
        top = 0;
    }


    void destructor(){
        delete[] mas;
        init();
    }
};
