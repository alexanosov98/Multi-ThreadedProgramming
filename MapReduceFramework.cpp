#include "MapReduceFramework.h"
#include <pthread.h>
#include "./Barrier/Barrier.h"
#include <atomic>
#include <iostream>
#include <algorithm>


#define LAST_TWO_BITS 0x03
#define THIRTY_ONE_BITS 0x7FFFFFFF
#define STAGE_SHIFT 62
#define PROCESSED_SHIFT 31
#define STAGE_BITS >> STAGE_SHIFT & LAST_TWO_BITS // todo might be wrong
#define PROCESSED_BITS >> PROCESSED_SHIFT & THIRTY_ONE_BITS
#define TOTAL_BITS & THIRTY_ONE_BITS

typedef std::vector<IntermediateVec*>* twoDVec;




struct Job{
    const MapReduceClient* mapReduceClient;
    int multiThreadLevel;
    JobState state;
    std::atomic<uint64_t>* jobStateCounter;
    pthread_t* pThreads;
    const InputVec* inputVec;
    OutputVec* outputVec;
    twoDVec sortedVectors;
    twoDVec shuffledVec;
    Barrier* barrier;
    pthread_mutex_t* mutex;
    std::atomic<bool>* joined;
    std::atomic<int>* mapCounter;
    std::atomic<int>* sortShuffleCounter;
    std::atomic<int>* shuffledVecSizeCounter;
    std::atomic<int>* reduceCounter;

    ~Job(){
      delete[] pThreads;
      delete barrier;
      pthread_mutex_destroy (mutex); //todo might need to handle better
      delete mapCounter;
      delete sortShuffleCounter;
      delete shuffledVecSizeCounter;
      delete reduceCounter;
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


bool pairCompareFunc(IntermediatePair a, IntermediatePair b){
  return a.first < b.first;
}


bool keysAreEqual(K2* a, K2* b){
  return !(a>b) && !(b>a);
}


void reduce(ThreadContext* threadContext){
  Job* job = threadContext->job;
  //job->state = REDUCE_STAGE;
  std::atomic<int>* reduceCounter = job->reduceCounter;
  const MapReduceClient* client = job->mapReduceClient;
  while(reduceCounter->load() < job->shuffledVecSizeCounter->load ()){
    int oldVal = *reduceCounter++;
    client->reduce (job->shuffledVec->at(oldVal), threadContext);
  }
}


void shuffle(ThreadContext* threadContext)
{
  twoDVec sortedVectors = threadContext->job->sortedVectors;
  std::atomic<int> *sortShuffleCounter = threadContext->job->sortShuffleCounter;
  //todo check if there is an empty vector!
  while (sortShuffleCounter->load () > 0)
  {
    K2 *maxKey = nullptr;
    int maxKeyVecIdx = -1;
    //Find the biggest key currently available
    for (int i = 0; i < sortShuffleCounter->load(); i++)
    {
      IntermediateVec *currentVec = sortedVectors->at (i);

      if (currentVec->empty()){
        sortedVectors->erase(sortedVectors->begin() + i);
        sortShuffleCounter--;
        break;
      }
      if (maxKeyVecIdx == -1)
      {
        maxKeyVecIdx = i;
        maxKey = currentVec->back().first;
      }
      else
      {
        if (maxKey < currentVec->back().first)
        {
          maxKeyVecIdx = i;
          maxKey = currentVec->back().first;
        }
      }
    }
    //Create a vector that contains all the pairs with the max key
    auto newVec = new IntermediateVec ();
    for (auto sortedVec : *sortedVectors){
      while(keysAreEqual (sortedVec->back().first, maxKey) &&
      !sortedVec->empty()){
        newVec->push_back (sortedVec->back());
        sortedVec->pop_back();
      }
    }
    threadContext->job->shuffledVec->push_back(newVec);
    threadContext->job->shuffledVecSizeCounter++;
  }
  reduce(threadContext);
}


void sort(ThreadContext* threadContext){
  //Sort
  //todo check if there is an empty vector
  std::sort (threadContext->intermediateVec->begin (),
             threadContext->intermediateVec->end (), pairCompareFunc);
  //Add the sorted vector to the shared pile
  if (pthread_mutex_lock(threadContext->job->mutex) != 0){
    std::cerr << "system error: Failed to lock mutex" << std::endl;
    exit(1);
  }
  threadContext->job->sortedVectors->push_back
  (threadContext->intermediateVec);
  threadContext->job->sortShuffleCounter++;
  threadContext->job->jobStateCounter++;
  if (pthread_mutex_unlock(threadContext->job->mutex) != 0){
    std::cerr << "system error: Failed to unlock mutex" << std::endl;
    exit(1);
  }
}


void map(ThreadContext* threadContext){
  Job* job = threadContext->job;
  std::atomic<int>* mapStageCounter = job->mapCounter;
    const MapReduceClient* client = job->mapReduceClient;

    while(mapStageCounter->load() < (int)job->inputVec->size()){ //todo size
      // might be bad
      int oldVal = *mapStageCounter++;
      auto key = job->inputVec->at(oldVal).first;
      auto val = job->inputVec->at(oldVal).second;
      client->map (key,val, threadContext);
      job->jobStateCounter++;
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
      (static_cast<uint64_t>(0) << PROCESSED_SHIFT) | static_cast<uint64_t>
      (total);
  tc->job->jobStateCounter->store(newCounterVal);
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
  if (threadContext->Id == 0){
    updateJobState(threadContext, threadContext->job->inputVec->size(),
                   MAP_STAGE);
  }
  map(threadContext);

  threadContext->job->barrier->barrier();
  if (threadContext->Id == 0){
    //Find total of elements to shuffle
    uint32_t shuffleTotal = 0;
    for (int i = 0; i < (int)threadContext->job->sortedVectors->size(); i++){
      shuffleTotal += threadContext->job->sortedVectors->at(i)->size();
    }
    updateJobState (threadContext,shuffleTotal, SHUFFLE_STAGE);
    shuffle(threadContext);
  }
  threadContext->job->barrier->barrier();
  uint32_t reduceTotal = 0;
  for (int i = 0; i < (int)threadContext->job->shuffledVec->size(); i++){
    reduceTotal += threadContext->job->shuffledVec->at(i)->size();
  }
  updateJobState (threadContext, reduceTotal, REDUCE_STAGE); //todo i think
  // reduce total is the same as shuffle total
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
  if (pthread_mutex_init(job->mutex, nullptr) != 0){
      std::cerr << "system error: Failed to create new mutexes" << std::endl;
      exit(1);
    }
  //Create atomic counters and find its max value
  job->mapCounter = new std::atomic<int> (0);
  job->sortShuffleCounter = new std::atomic<int> (0);
  job->shuffledVecSizeCounter = new std::atomic<int> (0);
  job->reduceCounter = new std::atomic<int> (0);
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
    if (pthread_create (thread, nullptr, threadRoutine,
                        thread))
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
    if(!jc->job->joined->load ()){
      jc->job->joined->store(true);
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
  auto counterVal = jc->job->jobStateCounter->load();
  //Extract data via shifting the 64 bit unsigned long int
  auto total = counterVal TOTAL_BITS;
  auto processed = counterVal PROCESSED_BITS;
  auto stage = counterVal STAGE_BITS;
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
  ((ThreadContext*) context)->intermediateVec->push_back (std::pair<K2*, V2*>
      (key, value));
}


void emit3 (K3* key, V3* value, void* context){
  auto* tc = (ThreadContext*) context;
  //Lock the mutex to access the shared output vector
  if (pthread_mutex_lock(tc->job->mutex) != 0){
    std::cerr << "system error: Failed to lock mutex" << std::endl;
    exit(1);
  }

  std::pair<K3*, V3*> newPair(key, value);
  tc->job->outputVec->push_back (newPair);

  if (pthread_mutex_unlock(tc->job->mutex) != 0){
    std::cerr << "system error: Failed to unlock mutex" << std::endl;
    exit(1);
  }
}
