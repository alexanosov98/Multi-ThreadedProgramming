#include "MapReduceFramework.h"
#include <pthread.h>
#include "./Barrier/Barrier.h"
#include <atomic>
#include <iostream>
#include <algorithm>


#define LAST_TWO_BITS 0x03
#define THIRTY_ONE_BITS 0x7FFFFFFF
#define STAGE_SHIFT 62
#define TOTAL_SHIFT 31
#define STAGE_BITS(x) ((x >> STAGE_SHIFT) & LAST_TWO_BITS)
#define PROCESSED_BITS(x) (x & THIRTY_ONE_BITS)
#define TOTAL_BITS(x) ((x >> TOTAL_SHIFT) & THIRTY_ONE_BITS)


typedef std::vector<IntermediateVec*>* twoDVec;


struct Job{
    const MapReduceClient* mapReduceClient;
    int multiThreadLevel;
    JobState state;
    std::atomic<uint64_t> jobStateCounter;
    std::atomic<int> sortCounter;
    std::atomic<bool> joined;
    pthread_t* pThreads;
    const InputVec* inputVec;
    OutputVec* outputVec;
    twoDVec sortedVectors;
    twoDVec shuffledVec;
    Barrier* barrier;
    pthread_mutex_t* mutex;

    ~Job(){
      delete[] pThreads;
      delete barrier;
      pthread_mutex_destroy (mutex); //todo might need to handle better
    }
};


struct ThreadContext{
    int Id;
    Job* job;
    IntermediateVec* intermediateVec;
};


struct JobContext{
    Job* job;
    stage_t state;
    ThreadContext* threadContexts;
};


bool pairCompareFunc(IntermediatePair& a, IntermediatePair& b){
  return *(a.first) < *(b.first);
}


bool keysAreEqual(K2& a, K2& b){
  return !(a<b) && !(b<a);
}


void reduce(ThreadContext* threadContext){
  Job* job = threadContext->job;
  const MapReduceClient* client = job->mapReduceClient;

  if (pthread_mutex_lock (threadContext->job->mutex) != 0){
      std::cerr << "system error: Failed to lock mutex" << std::endl;
      exit(1);
  }
  auto total = TOTAL_BITS(job->jobStateCounter.load());
  auto processed = PROCESSED_BITS(job->jobStateCounter.load());
  if (pthread_mutex_unlock (threadContext->job->mutex) != 0){
      std::cerr << "system error: Failed to unlock mutex" << std::endl;
      exit(1);
  }

  while(processed < total){
      if (pthread_mutex_lock (threadContext->job->mutex) != 0){
          std::cerr << "system error: Failed to lock mutex" << std::endl;
          exit(1);
      }
      int oldVal = processed;
      job->jobStateCounter.fetch_add(1);
      processed = PROCESSED_BITS(job->jobStateCounter.load());
      if (pthread_mutex_unlock (threadContext->job->mutex) != 0){
          std::cerr << "system error: Failed to unlock mutex" << std::endl;
          exit(1);
      }
      client->reduce (job->shuffledVec->at(oldVal), threadContext);
  }
}


void shuffle(ThreadContext* threadContext) {
    twoDVec sortedVectors = threadContext->job->sortedVectors;
    auto job = threadContext->job;
    auto total = TOTAL_BITS(job->jobStateCounter.load());
    auto processed = PROCESSED_BITS(job->jobStateCounter.load());
    bool createNewVec = true;
    //todo check if there is an empty vector!
    while (processed < total) { //todo notice counter increment
        K2 *maxKey = nullptr;
        int maxKeyVecIdx = -1;
        //Find the biggest key currently available
        for (int i = 0; i < job->sortCounter.load(); i++) {
            IntermediateVec *currentVec = sortedVectors->at(i);

            if (currentVec->empty()) {
                sortedVectors->erase(sortedVectors->begin() + i);
                job->sortCounter.fetch_add(-1);
                createNewVec = false;
                break;
            }
            if (maxKeyVecIdx == -1) {
                maxKeyVecIdx = i;
                maxKey = currentVec->back().first;
            } else {
                if (maxKey < currentVec->back().first) {
                    maxKeyVecIdx = i;
                    maxKey = currentVec->back().first;
                }
            }
        }
        if (createNewVec) {
            //Create a vector that contains all the pairs with the max key
            auto newVec = new IntermediateVec();
            for (auto sortedVec: *sortedVectors) {
                while (!sortedVec->empty() && keysAreEqual(*sortedVec->back().first, *maxKey)) {
                    newVec->push_back(sortedVec->back());
                    job->jobStateCounter.fetch_add(1);
                    processed = job->jobStateCounter.load() & THIRTY_ONE_BITS;
                    sortedVec->pop_back();
                }
            }
            threadContext->job->shuffledVec->push_back(newVec);
        }
        createNewVec = true;
    }
}


void sort(ThreadContext* threadContext){
  std::sort (threadContext->intermediateVec->begin (),
             threadContext->intermediateVec->end (), pairCompareFunc);

  if (pthread_mutex_lock(threadContext->job->mutex) != 0){
    std::cerr << "system error: Failed to lock mutex" << std::endl;
    exit(1);
  }

  threadContext->job->sortedVectors->push_back(threadContext->intermediateVec);
  threadContext->job->sortCounter.fetch_add(1);

  if (pthread_mutex_unlock(threadContext->job->mutex) != 0){
    std::cerr << "system error: Failed to unlock mutex" << std::endl;
    exit(1);
  }
}


void map(ThreadContext* threadContext) {
    if (pthread_mutex_lock (threadContext->job->mutex) != 0){
        std::cerr << "system error: Failed to lock mutex" << std::endl;
        exit(1);
    }
    auto job = threadContext->job;
    const MapReduceClient* client = job->mapReduceClient;
    auto total = TOTAL_BITS(job->jobStateCounter.load());
    auto processed = PROCESSED_BITS(job->jobStateCounter.load());
    if (pthread_mutex_unlock (threadContext->job->mutex) != 0){
        std::cerr << "system error: Failed to unlock mutex" << std::endl;
        exit(1);
    }

    while (true) {
        if (pthread_mutex_lock (threadContext->job->mutex) != 0){
            std::cerr << "system error: Failed to lock mutex" << std::endl;
            exit(1);
        }
        if (processed >= total){
            if (pthread_mutex_unlock (threadContext->job->mutex) != 0){
                std::cerr << "system error: Failed to unlock mutex" << std::endl;
                exit(1);
            }
            break;
        }
        int oldVal = processed;
        InputPair pair = job->inputVec->at(oldVal);
        job->jobStateCounter.fetch_add(1);
        if (pthread_mutex_unlock (threadContext->job->mutex) != 0){
            std::cerr << "system error: Failed to unlock mutex" << std::endl;
            exit(1);
        }
        client->map(pair.first, pair.second, threadContext);
//
//        int oldVal = job->jobStateCounter.fetch_add(1)& THIRTY_ONE_BITS;
//        InputPair pair = job->inputVec->at(oldVal);
//        client->map(pair.first, pair.second, threadContext);
//
//        processed = PROCESSED_BITS(job->jobStateCounter.load());
    }
    sort(threadContext);
}


void updateJobState(ThreadContext* tc, uint32_t total, stage_t stage){
  if (pthread_mutex_lock (tc->job->mutex) != 0){
    std::cerr << "system error: Failed to lock mutex" << std::endl;
    exit(1);
  }
  tc->job->state.percentage = 0;
  tc->job->state.stage = stage;
  uint64_t newCounterVal = (static_cast<uint64_t>(stage) << STAGE_SHIFT) |
      (static_cast<uint64_t>(total) << TOTAL_SHIFT) | static_cast<uint64_t>
      (0);
  tc->job->jobStateCounter.store(newCounterVal);
  if (pthread_mutex_unlock (tc->job->mutex) != 0){
    std::cerr << "system error: Failed to unlock mutex" << std::endl;
    exit(1);
  }
}


/*
 * Puts the whole Map, Reduce, Shuffle, Sort of the client into action on
 * the thread level.
 */
void* threadRoutine(void* tc){
  auto threadContext = (ThreadContext*)tc;
  //Map and Sort stage
  threadContext->job->barrier->barrier();
  if (threadContext->Id == 0){
    updateJobState(threadContext, threadContext->job->inputVec->size(),
                   MAP_STAGE);
  }
  threadContext->job->barrier->barrier();

  threadContext->job->barrier->barrier();
  map(threadContext);
  threadContext->job->barrier->barrier();

  //Shuffle stage
  threadContext->job->barrier->barrier();
  if (threadContext->Id == 0){
    uint32_t shuffleTotal = 0;
    for (int i = 0; i < (int)threadContext->job->sortedVectors->size(); i++){
      shuffleTotal += threadContext->job->sortedVectors->at(i)->size();
    }
    updateJobState (threadContext,shuffleTotal, SHUFFLE_STAGE);
    shuffle(threadContext);
  }
  threadContext->job->barrier->barrier();

  //Reduce stage
  threadContext->job->barrier->barrier();
  if (threadContext->Id == 0){
      uint32_t reduceTotal = threadContext->job->shuffledVec->size();
      updateJobState (threadContext, reduceTotal, REDUCE_STAGE);
  }
  threadContext->job->barrier->barrier();
  reduce(threadContext);

  return (void*) threadContext;
}


JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){

  //Create new Job
  Job* job = new Job();
  if(job == nullptr){
    std::cerr << "system error: Failed to create new job" << std::endl;
    exit(1);
  }
  //Add input arguments to new job
  job->outputVec = &outputVec;
  job->inputVec = &inputVec;
  job->mapReduceClient = &client;
  job->multiThreadLevel = multiThreadLevel;
  job->state = {UNDEFINED_STAGE, 0};

  job->sortedVectors = new std::vector<IntermediateVec*>;
  job->shuffledVec = new std::vector<IntermediateVec*>;
  job->sortCounter.store(0);

  JobContext* jobContext = new JobContext;
  jobContext->state = UNDEFINED_STAGE;
  jobContext->job = job;
  if(jobContext == nullptr){
    std::cerr << "system error: Failed to create new job context" << std::endl;
    exit(1);
  }
  //Create new Barrier
  Barrier* barrier = new Barrier(multiThreadLevel);
  if(barrier == nullptr){
    std::cerr << "system error: Failed to create new barrier" << std::endl;
    exit(1);
  }
  job->barrier = barrier;
  //Create Mutex
  job->mutex = new pthread_mutex_t;
  if (pthread_mutex_init(job->mutex, nullptr) != 0){
      std::cerr << "system error: Failed to create new mutexes" << std::endl;
      exit(1);
    }
  job->joined = new std::atomic<bool>(false);
  //Create thread contexts
  jobContext->threadContexts = new ThreadContext[multiThreadLevel];
  if(jobContext->threadContexts == nullptr){
    std::cerr << "system error: Failed to create new thread contexts" <<
    std::endl;
    exit(1);
  }
  for (int i = 0; i < multiThreadLevel; i++){
    jobContext->threadContexts[i].Id = i;
    jobContext->threadContexts[i].job = job;
    jobContext->threadContexts[i].intermediateVec = new IntermediateVec;
  }
  //Create pthreads
  job->pThreads = new pthread_t [multiThreadLevel];
  if (job->pThreads == nullptr){
    std::cerr << "system error: Failed to create new pthread array" <<
              std::endl;
    exit(1);
  }
  for (int i = 0 ; i < multiThreadLevel; i++)
  {
    pthread_t *thread = job->pThreads + i;
    ThreadContext* tc = jobContext->threadContexts + i;
    if (pthread_create (thread, nullptr, threadRoutine,
                        tc))
    {
      std::cout << "system error: Failed to create new pthread" << std::endl;
      exit (1);
    }
  }
  return (JobHandle) jobContext;
}


void waitForJob(JobHandle job){
    auto* jc = (JobContext*)job;
    if (pthread_mutex_lock (jc->job->mutex) != 0){
      std::cerr << "system error: Failed to lock mutex" << std::endl;
      exit(1);
    }
    if(!jc->job->joined.load ()){
      jc->job->joined.store(true);
      for (int i = 0; i < jc->job->multiThreadLevel ; i++){
        if (pthread_join (*(jc->job->pThreads + i), nullptr) != 0){
          std::cerr << "system error: Failed to join thread" << std::endl;
          exit(1);
        }
      }
    }
  if (pthread_mutex_unlock (jc->job->mutex) != 0){
    std::cerr << "system error: Failed to unlock mutex" << std::endl;
    exit(1);
  }
}


void getJobState(JobHandle job, JobState* state){
  auto jc = (JobContext*) job;
  auto counterVal = jc->job->jobStateCounter.load();
  //Extract data via shifting the 64 bit unsigned long int
  auto total = TOTAL_BITS(counterVal);
  auto processed = PROCESSED_BITS(counterVal);
  auto stage = STAGE_BITS(counterVal);
  //Update the data in the argument
  state->stage = static_cast<stage_t>(stage);
  state->percentage = (float(processed) /(float) total) * 100;
}


void closeJobHandle(JobHandle job){
  auto* jc = static_cast<JobContext*>(job);
  waitForJob (job);
  delete jc;
}


void emit2 (K2* key, V2* value, void* context){
    ThreadContext* tc = (ThreadContext*) context;
    if (pthread_mutex_lock (tc->job->mutex) != 0){
        std::cerr << "system error: Failed to lock mutex" << std::endl;
        exit(1);
    }

    tc->intermediateVec->push_back (std::pair<K2*, V2*>(key, value));

    if (pthread_mutex_unlock (tc->job->mutex) != 0){
        std::cerr << "system error: Failed to unlock mutex" << std::endl;
        exit(1);
    }
}


void emit3(K3* key, V3* value, void* context) {
    auto* tc = (ThreadContext*) context;
    // Lock the mutex to access the shared output vector
    if (pthread_mutex_lock(tc->job->mutex) != 0){
        std::cerr << "system error: Failed to lock mutex" << std::endl;
        exit(1);
    }

    std::pair<K3*, V3*> newPair(key, value);
    tc->job->outputVec->push_back(newPair);

    if (pthread_mutex_unlock(tc->job->mutex) != 0){
        std::cerr << "system error: Failed to unlock mutex" << std::endl;
        exit(1);
    }
}

