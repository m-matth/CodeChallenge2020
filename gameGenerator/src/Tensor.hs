{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}

module Tensor where

import Control.Monad.State
import Control.Lens.Getter ((^.))
import Control.Lens.Fold ((^..))
import Control.Lens.Combinators (traversed)
import Control.Lens.Operators ((.~))
import qualified Data.Array as A
import Data.Int (Int64)
import Data.Maybe (catMaybes)
import Data.Vector (Vector)
import Data.Functor.Identity (runIdentity)
import qualified Data.Text as T
import qualified Data.Vector as V
import qualified System.Random as Rand
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Ops as TF2
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Variable as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields   as TF (node)
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields as TF (name, op, value)
import qualified TensorFlow.Build as TF
import Debug.Trace
import Environment
import GHC.Float
import qualified Data.ByteString.Char8 as B8

import Lib as Game

import TensorflowContrib as TFContrib

charToTile :: Char -> Game.Cell
charToTile '?' = Unknow
charToTile '#' = Wall
charToTile '$' = Enemy
charToTile '@' = Opponent
charToTile '*' = Me
charToTile '!' = Shadow
charToTile '/' = FiredUp
charToTile '\\' = FiredDown
charToTile '>' = FiredLeft
charToTile '<' = FiredRight
charToTile ' ' = Empty

type GameObservation = Word

data GameEnvironment = GameEnvironment
  { currentObs :: GameObservation
  , grid :: A.Array Word Game.Cell
  , slipChance :: Double -- 0 to 1
  , randomGenerator :: Rand.StdGen
  , previousAction :: Maybe Game.Action
  , dimens :: (Word, Word) -- Rows, Cols
  , flExplorationRate :: Double
  , totalScore :: Int
  }

-- Build Model
-- Choose Action
-- Learn from the environment

data Model = Model
  { weightsT :: TF.Variable Float
  , chooseActionStep :: TF.TensorData Float -> TF.Session (Vector Float)
  , learnStep :: TF.TensorData Float -> TF.TensorData Float -> TF.Session ()


  , inputLayerName :: T.Text
  , outputLayerName :: T.Text
  , labelLayerName :: T.Text
  , neuralNetworkVariables :: [TF.Tensor TF.Ref Float] -- ^ Neural network variables for saving and restoring.
  , trainingVariables      :: [TF.Tensor TF.Ref Float] -- ^ Training data/settings for saving and restoring.
  }


modelBuilder :: (TF.MonadBuild m) => m Model
modelBuilder = do
  let x = 10
      y = 10
      boardSize = fromIntegral $ x*y
      nbOfAction = fromIntegral Game.nbOfAction

  -- Choose Action
  let inputLayerName_ = T.pack "input"
  --  (inputs :: TF.Tensor TF.Value Float) <- TF.placeholder (TF.Shape [1, boardSize])
  (inputs :: TF.Tensor TF.Value Float) <- TF.placeholder' (TF.opName .~ TF.explicitName inputLayerName_) (TF.Shape [1, boardSize])

  let outputLayerName_ = T.pack "ouput"
  (weights :: TF.Variable Float) <- TF.truncatedNormal (TF.vector [boardSize, nbOfAction]) >>= TF2.initializedVariable' (TF.opName .~ "w1")
--  weights_ <- TF.initializedVariable' (TF.opName .~ "w1") =<< randomParam numInputs [numInputs, numUnits]
  
  let (results :: TF.Tensor TF.Build Float) = (inputs `TF.matMul` TF.readValue weights)
  --  (returnedOutputs :: TF.Tensor TF.Value Float) <- TF.render results
  (returnedOutputs :: TF.Tensor TF.Value Float) <- TF.render $ TF.identity' (TF.opName .~ TF.explicitName outputLayerName_) $ results

  -- Train Network
  let labelLayerName_ = T.pack "label"
  --  (nextOutputs :: TF.Tensor TF.Value Float) <- TF.placeholder (TF.Shape [nbOfAction, 1])
  (nextOutputs :: TF.Tensor TF.Value Float) <- TF.placeholder' (TF.opName .~ TF.explicitName labelLayerName_) (TF.Shape [nbOfAction, 1])

  -- Data Collection
  let weights_ = [weights_] :: [TF.Tensor TF.Ref Float]
  let (diff :: TF.Tensor TF.Build Float) = nextOutputs `TF.sub` results
  let (loss :: TF.Tensor TF.Build Float) = TF.reduceSum (diff `TF.mul` diff)
  trainer_ <- TF.minimizeWith TF.adam loss [weights]
  let chooseStep = \inputFeed -> TF.runWithFeeds [TF.feed inputs inputFeed] returnedOutputs
  let trainStep = \inputFeed nextOutputFeed ->
        TF.runWithFeeds [TF.feed inputs inputFeed, TF.feed nextOutputs nextOutputFeed] trainer_

  return $ Model weights chooseStep trainStep inputLayerName_ outputLayerName_ labelLayerName_ weights_ []

  
  
createModel :: Game.Width -> Game.Height -> TF.Session Model
createModel x y = do
  -- Choose Action
  let inputLayerName_ = T.pack "input"
  --  (inputs :: TF.Tensor TF.Value Float) <- TF.placeholder (TF.Shape [1, boardSize])
  (inputs :: TF.Tensor TF.Value Float) <- TF.placeholder' (TF.opName .~ TF.explicitName inputLayerName_) (TF.Shape [1, boardSize])

  let outputLayerName_ = T.pack "ouput"
  (weights :: TF.Variable Float) <- TF.truncatedNormal (TF.vector [boardSize, nbOfAction]) >>= TF.initializedVariable
  let (results :: TF.Tensor TF.Build Float) = (inputs `TF.matMul` TF.readValue weights)
  --  (returnedOutputs :: TF.Tensor TF.Value Float) <- TF.render results
  (returnedOutputs :: TF.Tensor TF.Value Float) <- TF.render $ TF.identity' (TF.opName .~ TF.explicitName outputLayerName_) $ results

  -- Train Network
  let labelLayerName_ = T.pack "label"
  --  (nextOutputs :: TF.Tensor TF.Value Float) <- TF.placeholder (TF.Shape [nbOfAction, 1])
  (nextOutputs :: TF.Tensor TF.Value Float) <- TF.placeholder' (TF.opName .~ TF.explicitName labelLayerName_) (TF.Shape [nbOfAction, 1])

  -- Data Collection
  let weights_ = [] -- [weights] :: [TF.Tensor TF.Ref Float]

  
  let (diff :: TF.Tensor TF.Build Float) = nextOutputs `TF.sub` results
  let (loss :: TF.Tensor TF.Build Float) = TF.reduceSum (diff `TF.mul` diff)
  trainer_ <- TF.minimizeWith TF.adam loss [weights]
  let chooseStep = \inputFeed -> TF.runWithFeeds [TF.feed inputs inputFeed] returnedOutputs
  let trainStep = \inputFeed nextOutputFeed ->
        TF.runWithFeeds [TF.feed inputs inputFeed, TF.feed nextOutputs nextOutputFeed] trainer_
  return $ Model weights chooseStep trainStep inputLayerName_ outputLayerName_ labelLayerName_ weights_ []
  where
    boardSize = fromIntegral $ x*y
    nbOfAction = fromIntegral Game.nbOfAction

reset :: ChallengeGame GameObservation
reset = ChallengeGame $ do
  let initialObservation = trace "ici" 0
  (fle, model) <- get
  put $ (fle { currentObs = initialObservation, previousAction = Nothing }, model)
  return initialObservation

step :: Game.Action -> ChallengeGame (GameObservation, Reward, Bool)
step act = do
  fle <- get
  let obs = currentObs fle
  let (slipRoll, gen') = Rand.randomR (0.0, 1.0) (randomGenerator fle)
  let allLegalMoves = legalMoves obs (dimens fle)
  let (randomMoveIndex, finalGen) = Rand.randomR (0, length allLegalMoves - 1) gen'
  let newObservation = if slipRoll >= slipChance fle
        then if act `elem` allLegalMoves
          then applyMoveUnbounded act obs (snd . dimens $ fle)
          else obs
        else applyMoveUnbounded (allLegalMoves !! randomMoveIndex) obs (snd . dimens $ fle)
  let (done, reward) = case (grid fle) A.! newObservation of
        FiredUp -> (True, Reward 0.0)
        FiredDown -> (True, Reward 0.0)
        FiredLeft -> (True, Reward 0.0)
        FiredRight -> (True, Reward 0.0)
        Enemy -> (True, Reward 0.0)
        Opponent -> (True, Reward 0.0)
        Wall -> (False, Reward 0.0)
        _ -> (False, Reward 1.0)
  let (Reward r) = reward
--  let newScore = trace ("score " ++ show (totalScore fle)) ((totalScore fle) + (double2Int r))
  let newScore = (totalScore fle) + (double2Int r)
  put $ fle {currentObs = newObservation, randomGenerator = finalGen, previousAction = Just act, totalScore = newScore}
  return (newObservation, reward, if newScore > 10 then True else done)
-- trace ("reward " ++ show reward) 
-- instance Show Reward where
--  show (Reward x) = show x

basicEnv :: IO GameEnvironment
basicEnv = do
  gen <- Rand.getStdGen
  board <- fmap fromBoard (generateInitialBoard Game.boardY Game.boardX)
  let _ = trace ("ici2 " ++ show board) board
  return $ GameEnvironment
    { currentObs = 0
    , grid = A.listArray (0, fromIntegral (sz - 1)) (charToTile <$>  board)--  "####### $ ###$  $## @#@#######")
    , slipChance = 0.0
    , randomGenerator = gen
    , previousAction = Nothing
    , dimens = (4, 4)
    , flExplorationRate = 0.9
    , totalScore = 0
    }
  where
    sz = Game.boardY * Game.boardX

obsToTensor :: GameObservation -> TF.TensorData Float
obsToTensor obs = TF.encodeTensorData (TF.Shape [1, fromIntegral boardSize]) (V.fromList asList)
  where
    asList = replicate (fromIntegral obs) 0.0 ++ [1.0] ++ replicate (fromIntegral ((boardSize - 1) - obs)) 0.0
    boardSize = fromIntegral $ Game.boardY * Game.boardX

chooseActionTensor :: ChallengeGame Game.Action
chooseActionTensor = ChallengeGame $ do
  (fle, model) <- get
  let (exploreRoll, gen') = Rand.randomR (0.0, 1.0) (randomGenerator fle)
  if exploreRoll < flExplorationRate fle
    then do
      let (actionRoll, gen'') = Rand.randomR (0, nbOfAction - 1) gen'
      put $ (fle { randomGenerator = gen'' }, model)
      return (toEnum $ fromIntegral actionRoll)
    else do
      let obs1 = currentObs fle
      let obs1Data = obsToTensor obs1
      (results :: Vector Float) <- lift ((chooseActionStep model) obs1Data)
      let bestMoveIndex = V.maxIndex results
      put $ (fle { randomGenerator = gen' }, model)
      return (toEnum bestMoveIndex)

learnTensor ::
  GameObservation -> GameObservation ->
  Reward -> Game.Action ->
  ChallengeGame ()
learnTensor obs1 obs2 (Reward reward) action = ChallengeGame $ do
  model <- snd <$> get
  let obs1Data = obsToTensor obs1
  (results :: Vector Float) <- lift ((chooseActionStep model) obs1Data)
  let (bestMoveIndex, maxScore) = (V.maxIndex results, V.maximum results)
  let targetActionValues = results V.// [(bestMoveIndex, double2Float reward + (gamma * maxScore))]
  let obs2Data = obsToTensor obs2
  let targetActionData = TF.encodeTensorData (TF.Shape [fromIntegral nbOfAction, 1]) targetActionValues
  lift $ (learnStep model) obs2Data targetActionData
  where
    gamma = 0.81

newtype ChallengeGame a = ChallengeGame
  (StateT (GameEnvironment, Model) TF.Session a)
  deriving (Functor, Applicative, Monad, MonadIO)

{--
instance TF.MonadBuild (StateT (GameEnvironment, Model) TF.Session) where
  build = TF.hoistBuildT $ return . runIdentity
--}
instance (MonadState GameEnvironment) ChallengeGame where
  get = ChallengeGame (fst <$> get)
  put fle = ChallengeGame $ do
    (_, model) <- get
    put (fle, model)

instance EnvironmentMonad ChallengeGame where
  type (Observation ChallengeGame) = GameObservation
  type (Action ChallengeGame) = Game.Action
  type (EnvironmentState ChallengeGame) = GameEnvironment
  baseEnv = basicEnv
  currentObservation = currentObs <$> get
  resetEnv = reset
  stepEnv = step
  runEnv env (ChallengeGame action) width height = TF.runSession $ do
    model <- createModel width height
    TF.save  (B8.pack "/tmp/foo/") (neuralNetworkVariables model) >>= TF.run_
    evalStateT action (env, model)
  saveEnv = save

instance LearningEnvironment ChallengeGame where
  chooseActionBrain = chooseActionTensor
  learnEnv = learnTensor
  explorationRate = flExplorationRate <$> get
  reduceExploration decayRate minEpsilon = do
    fle <- get
    let e = flExplorationRate fle
    let newE = max minEpsilon (e * decayRate)
    put $ fle { flExplorationRate = newE }

legalMoves :: GameObservation -> (Word, Word) -> [Game.Action]
legalMoves observation (numRows, numCols) = catMaybes [left, down, right, up, none]
  where
    (row, col) = quotRem observation numRows
    left = if col > 0 then Just MoveLeft else Nothing
    down = if row < numRows - 1 then Just MoveDown else Nothing
    right = if col < numCols - 1 then Just MoveRight else Nothing
    up = if row > 0 then Just MoveUp else Nothing
    none = Just NoOp
   --  def = [ FireUp, FireDown, FireLeft, FireRight ]

-- Does NOT do bounds checking
applyMoveUnbounded :: Game.Action -> GameObservation -> Word -> GameObservation
applyMoveUnbounded action obs numCols = case action of
  MoveLeft -> obs - 1 
  MoveDown -> obs + numCols
  MoveRight -> obs + 1
  MoveUp -> obs - numCols
  NoOp -> obs
{--  FireUp -> obs
  FireDown -> obs
  FireLeft -> obs
  FireRight -> obs
--}

-- save :: B8.ByteString -> B8.ByteString -> Int -> ChallengeGame ()
--save pathModel pathTrain i = do

save = undefined
{--
save :: B8.ByteString -> B8.ByteString -> Int -> ChallengeGame ()
save pathModel pathTrain i = ChallengeGame $ do
  (fle, model) <- get
--  let obs1 = currentObs fle
--  let obs = currentObservation
--  let obs1Data = obsToTensor obs1
--  (inputLayerName_ :: T.Text) <- lift (inputLayerName model)
  let inputLayerName_ = inputLayerName model :: T.Text

  let graphDef = TF.asGraphDef modelBuilder
      namesPredictor = graphDef ^.. TF.node . traversed . TF.name

  let outputTensor = head (graphDef ^. TF.node)
      outputTensorName = outputTensor ^. TF.name
      inputTensorName = last (graphDef ^. TF.node)^. TF.name

  let inRef = TF.tensorFromName (inputLayerName model) :: TF.Tensor TF.Ref Float
      outRef = TF.tensorFromName (outputLayerName model) :: TF.Tensor TF.Ref Float
      labRef = TF.tensorFromName (labelLayerName model) :: TF.Tensor TF.Ref Float

  foo <- TF.save pathModel (neuralNetworkVariables model)
  TF.run_ foo
--  pure $ TF.save pathModel (neuralNetworkVariables model) >>= TF.run_
 --  pure $ TF.save pathTrain (trainingVariables model) >>= TF.runWithFeeds_ []

  return ()
--}


{--
save = do
  let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, 2] (V.fromList $ mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (V.fromList xs)

  tempDir <- getCanonicalTemporaryDirectory >>= flip createTempDirectory ""
  print $ "TempDir: " ++ tempDir
  let pathModel = B8.pack $ tempDir ++ "/model"
      pathTrain = B8.pack $ tempDir ++ "/train"

  let graphDef = TF.asGraphDef modelBuilder
      namesPredictor = graphDef ^.. TF.node.traversed.TF.name

  let outputTensor = head (graphDef ^. TF.node)
      outputTensorName = outputTensor ^. TF.name
      inputTensorName = last (graphDef ^. TF.node)^. TF.name

  let inp = encodeImageBatch [[0.7 :: Float,0.4]]
      lab = encodeLabelBatch [2.3 :: Float]
--}

{--
-- SESSION 1
TF.runSession $ do
  model <- modelBuilder
  let inRef = TF.tensorFromName (inputLayerName model) :: TF.Tensor TF.Ref Float
      outRef = TF.tensorFromName (outputLayerName model) :: TF.Tensor TF.Ref Float
      labRef = TF.tensorFromName (labelLayerName model) :: TF.Tensor TF.Ref Float

  bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
  liftIO $ putStrLn $ "START SESS 1: " ++ show bef

  forM_ ([0..1000] :: [Int]) $ \i -> do
    (x1Data :: [Float]) <- liftIO $ replicateM 1 randomIO
    (x2Data :: [Float]) <- liftIO $ replicateM 1 randomIO
    let xData = [[x1,x2] | x1 <- x1Data, x2 <- x2Data ]
    let yData = map (\(x1:x2:_) -> x1 * 0.3 + x2 * 0.5) xData
    let inpTrain = encodeImageBatch xData
        labTrain = encodeLabelBatch yData
    TF.runWithFeeds_ [TF.feed inRef inpTrain, TF.feed labRef labTrain] (trainingNode model)

    when (i `mod` 100 == 0) $ do
      bef <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
      liftIO $ putStrLn $ "Value: " ++ show bef
      varVals :: [V.Vector Float] <- TF.run (neuralNetworkVariables model)
      liftIO $ putStrLn $ "Weights: " ++ show (V.toList <$> varVals)

  aft <- head . V.toList <$> TF.runWithFeeds [TF.feed inRef inp] outRef
  liftIO $ putStrLn $ "END SESS 1: " ++ show aft
  TF.save pathModel (neuralNetworkVariables model) >>= TF.run_
  varVals :: [V.Vector Float] <- TF.runWithFeeds [TF.feed inRef inp, TF.feed labRef lab] (neuralNetworkVariables model)
  liftIO $ putStrLn $ "SESS 1 Weights: " ++ show (V.toList <$> varVals)
  TF.save pathTrain (trainingVariables model) >>= TF.runWithFeeds_ [TF.feed inRef inp, TF.feed labRef lab]
--}
