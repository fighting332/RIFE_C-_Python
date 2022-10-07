
#include <Python.h>
using namespace std;

int main()
{
    Py_Initialize();              //Initialize the virtual environment of Python
    if (Py_IsInitialized())
    {
        PyObject* pModule = NULL;
        PyObject* pFunc = NULL;
        pModule = PyImport_ImportModule("inference_img");  //Input the name of Python script
        if (pModule)
        {
            pFunc = PyObject_GetAttrString(pModule, "getframes");   //Obtain the function
            PyEval_CallObject(pFunc, NULL);           //Execute the function
        }
        else
        {
            printf("module fail...\n");
        }
    }
    else
    {
        printf("initialize fail...\n");
    }
    Py_Finalize();
}