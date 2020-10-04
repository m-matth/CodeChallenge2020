{-- imported from github.com/jhb563/OpenGymHs --}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Environment where

import Control.Monad (forM, when)
import Control.Monad.IO.Class (MonadIO, liftIO)
import qualified Data.ByteString.Char8 as B8
import Debug.Trace
import System.IO.Temp

newtype Reward = Reward Double

class (Monad m) => EnvironmentMonad m where
  type Observation m :: *
  type Action m :: *
  type EnvironmentState m :: *
  baseEnv :: IO (EnvironmentState m)
  runEnv :: (EnvironmentState m) -> m a -> Int -> Int -> IO a
  currentObservation :: m (Observation m)
  resetEnv :: m (Observation m)
  stepEnv :: (Action m) -> m (Observation m, Reward, Bool)
  saveEnv :: B8.ByteString -> B8.ByteString -> Int -> m ()

class (EnvironmentMonad m) => LearningEnvironment m where
  learnEnv :: (Observation m) -> (Observation m) -> Reward -> (Action m) -> m ()
  chooseActionBrain :: m (Action m)
  explorationRate :: m Double
  reduceExploration :: Double -> Double -> m ()

class (MonadIO m, EnvironmentMonad m) => RenderableEnvironment m where
  renderEnv :: m ()

gameLoop :: (EnvironmentMonad m) => m (Action m) -> m (Observation m, Reward)
gameLoop chooseAction = do
  newAction <- chooseAction
  (newObs, reward, done) <- stepEnv newAction
  if done
    then return (newObs, reward)
    else gameLoop chooseAction

gameLearningLoop :: (LearningEnvironment m) => m (Observation m, Reward)
gameLearningLoop = do
  oldObs <- currentObservation
  newAction <- chooseActionBrain
  (newObs, reward, done) <- stepEnv newAction
  learnEnv oldObs newObs reward newAction
  if done
    then return (newObs, reward)
    else gameLearningLoop

instance Show Reward where
  show (Reward x) = show x

gameLearningIterations :: (LearningEnvironment m, MonadIO m) => m [Reward]
gameLearningIterations =
  forM [1..numEpisodes] $ \i -> do
    liftIO $ putStrLn $ "iteration : " ++ show i
    resetEnv

--    when (i `mod` 10000 == 9999) $ do
--      liftIO $ putStrLn $ "saving !"
--      tempDir <- liftIO (getCanonicalTemporaryDirectory >>= flip createTempDirectory "")
--      let pathModel = B8.pack $ tempDir ++ "/model"
--          pathTrain = B8.pack $ tempDir ++ "/train"
--      saveEnv2 pathModel pathTrain i
--      liftIO $ putStrLn $ "saved !"
    when (i `mod` 100 == 99) $ do
      reduceExploration decayRate minEpsilon
    (_, reward) <- gameLearningLoop
    return (trace (show i ++ " reward = " ++ show reward) reward)
  where
    numEpisodes = 10000
    decayRate = 0.9
    minEpsilon = 0.01

gameRenderLoop :: (RenderableEnvironment m) => m (Action m) -> m (Observation m, Reward)
gameRenderLoop chooseAction = do
  renderEnv
  newAction <- chooseAction
  (newObs, reward, done) <- stepEnv newAction
  if done
    then renderEnv >> return (newObs, reward)
    else gameRenderLoop chooseAction
