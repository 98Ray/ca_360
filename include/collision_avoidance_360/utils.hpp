//
// Created by bruce on 2020/11/27.
//

#pragma once

#include <unistd.h>

using namespace std;

int isExisting(const string &filePath)
{
    return access(filePath.c_str(), 0) == 0;
}

class Logger
{
public:
    void info_once(const string &msg)
    {
        if (msg != _lastInfoMsg)
        {
            _lastInfoMsg = msg;
            ROS_INFO_STREAM(msg);
        }
    }

    void warn_once(const string &msg)
    {
        if (msg != _lastWarnMsg)
        {
            _lastWarnMsg = msg;
            ROS_WARN_STREAM(msg);
        }
    }

    void error_once(const string &msg)
    {
        if (msg != _lastErrorMsg)
        {
            _lastErrorMsg = msg;
            ROS_ERROR_STREAM(msg);
        }
    }

private:
    string _lastInfoMsg;
    string _lastWarnMsg;
    string _lastErrorMsg;
};