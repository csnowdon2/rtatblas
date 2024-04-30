//#include <matrixop.h>
//#include <timer_bank.h>
//
//template<typename Params, typename Key, typename Opts>
//class Executor {
//public:
//  typedef Params Params_T;
//  typedef Key    Key_T;
//  typedef Opts   Opts_T;
//
//  Executor() = default;
//  virtual ~Executor() = default;
//
//  virtual void execute(Params params, Opts opts, 
//                       Workspace space, Stream s, 
//                       Device_Timer::Mode sync = Device_Timer::ASYNCHRONOUS) {
//
//    if (!warm) {
//      warmup(params, opts, s);
//      warm = true;
//    }
//
//    Device_Timer timer([&](const Stream &str) {
//      internal_execute(params, opts, space, str);
//    }, s, sync);
//
//    timer_log[params][opts].append(timer);
//  }
//
//
//  std::map<Key, std::map<Opts, Timer_Bank>>& get_timings() {return timer_log;}
//  std::map<Opts, Timer_Bank>& get_timings(Key key) {return timer_log[key];}
//
//  virtual size_t calculate_workspace(Params, Opts) = 0;
//protected:
//  virtual void internal_execute(Params, Opts, Workspace, Stream) = 0;
//  virtual void warmup(Params, Opts, Stream) = 0;
//  std::map<Key, std::map<Opts, Timer_Bank>> timer_log;  
//
//  bool warm = false;
//};
//
