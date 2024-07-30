#pragma once
#include <timer_bank.h>
#include <workspace.h>
#include <map>

namespace rtat {

template<typename Params, typename Key, typename Opts>
class Executor {
public:
  typedef Params Params_T;
  typedef Key    Key_T;
  typedef Opts   Opts_T;

  Executor() = default;
  virtual ~Executor() = default;

  virtual void execute(Params params, Opts opts, 
                       Workspace space, Stream s, 
                       Device_Timer::Mode sync = Device_Timer::ASYNCHRONOUS) {

    if (!warm) {
      warmup(params, opts, s);
      warm = true;
    }

    Device_Timer timer([&](const Stream &str) {
      internal_execute(params, opts, space, str);
    }, s, sync);

    timer_log[params][opts].append(timer);
  }


  std::map<Key, std::map<Opts, Timer_Bank>>& get_timings() 
    { return timer_log; }
  std::map<Opts, Timer_Bank>& get_timings(Key key) 
    { return timer_log[key]; }

  virtual size_t calculate_workspace(Params params, Opts opts) {
    auto operation = opts.form_operation(params);
    return operation->workspace_req_bytes();
  }
protected:
  virtual void internal_execute(Params params, Opts opts, Workspace space,
                        [[maybe_unused]] Stream s) {
    auto operation = opts.form_operation(params);
    if (operation->workspace_req_bytes() > space.size<char>()) {
      throw "internal_execute: Insufficient workspace";
    }
    operation->execute(params.handle, Workspace(), space);
  }
  virtual void warmup(Params, Opts, Stream) = 0;

  std::map<Key, std::map<Opts, Timer_Bank>> timer_log;  
  bool warm = false;
};

}
