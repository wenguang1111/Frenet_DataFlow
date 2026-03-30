# Recorder4Cpp
This is a tool to help you to record your data in cpp program with different data types. The data with different types will be saved in corresponding csv files. Currently the data type integer, float and double are supported. But you can modify the template in source file to make it support for different types easily. 

You can decided if you want to activate the code for recording or not by turning on/off the option in cmake configuration. The code use singleton pattern so that you dont need care the instance of recorder. But it's important to add the function to **writeDataToCSV**() to force the recorder to write the data to csv. This function is usually added at the end of your application.

You need download the repo in your local project and modify your CMakeLists.txt for adding some option. 

```cmake
option(USE_RECORDER "Use recorder" OFF)
set(recorder_file ${recorder_file})

if (USE_RECORDER)
  MESSAGE(STATUS "USE_RECORDER=ON")
  add_definitions(-DUSE_RECORDER)
  set(recorder_file ${recorder_file} Recorder4Cpp/recorder.h
  	Recorder4Cpp/recorder.cpp)
endif()

add_library(YourLib SHARED
            src/*
      		****
            ${recorder_file})
            
add_executable(YourExecutable SHARED
            src/*
      		****
            ${recorder_file})
```

To record the file, you need add following code as example after the lane that your data get calculated:

```c++
void function()
{
    int myParameter_int = 3;
	float myParameter_float = 3.1;
	double myParameter_double = 3.1;
	#ifdef USE_RECORDER
		Recorder::getInstance()->saveData<int>("myParameter_int", myParameter_int);
    	Recorder::getInstance()->saveData<float>("myParameter_float", myParameter_float);
		Recorder::getInstance()->saveData<double>("myParameter_double", myParameter_double);
	#endif  
}
```

And don't forget to add the function writeDataToCSV() and the end. Otherwise the data are only saved in RAM but not in csv file:

```c++
int main()
{
    function();
    
    #ifdef USE_RECORDER
		// write recorded data to csv file
    	Recorder::getInstance()->writeDataToCSV();
	#endif
}
```

