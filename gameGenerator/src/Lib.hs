{-# LANGUAGE FlexibleInstances, UndecidableInstances #-}
{-# LANGUAGE LambdaCase #-}

module Lib
    ( someFunc
    ) where

import System.Random
import Data.List as List
import Control.Monad (replicateM, forever)
import Control.Concurrent (threadDelay)
import Debug.Trace

type Width = Int
type Height = Int

type BoardSize = (Width, Height)

data Board = Board [[Cell]]

instance Show Board where
    show (Board (x : ys)) = show x ++ "\n" ++ show (Board ys)
    show (Board x) = ""

data BoardAction = BoardAction [[Action]]

instance Show BoardAction where
    show (BoardAction (x : ys)) = show x ++ "\n" ++ show (BoardAction ys)
    show (BoardAction x) = ""

data Cell = Unknow
           | Wall
           | Enemy
           | Opponent
           | Me
           | Shadow
           | FiredUp
           | FiredDown
           | FiredLeft
           | FiredRight
           | Empty
           deriving ( Eq, Enum, Bounded)

data Action = MoveUp
            | MoveDown
            | MoveRight
            | MoveLeft
            | FireUp
            | FireDown
            | FireLeft
            | FireRight
            | NoOp
  deriving ( Eq, Show)

instance Show Cell where
  show Unknow =  "?"
  show Wall =  "#"
  show Enemy = "$"
  show Opponent = "@"
  show Me = "*"
  show Shadow = "!"
  show FiredUp = "/"
  show FiredDown = "\\"
  show FiredLeft = ">"
  show FiredRight = "<"
  show Empty = " "

instance {-# OVERLAPPABLE #-} (Bounded a, Enum a) => Random a where
   randomR (lo,hi) g = (toEnum i, g')
       where (i,g') = randomR (fromEnum lo, fromEnum hi) g
   random = randomR (minBound,maxBound)

defaultEnumRandomR :: (Enum a, RandomGen g) => (a, a) -> g -> (a, g)
defaultEnumRandomR (lo,hi) g = (toEnum i, g')
       where (i,g') = randomR (fromEnum lo, fromEnum hi) g

defaultBoundedRandom :: (Random a, Bounded a, RandomGen g) => g -> (a, g)
defaultBoundedRandom = randomR (minBound, maxBound)


instance Random Cell
  where
    randomR = defaultEnumRandomR
    random = defaultBoundedRandom


generateInitialBoard :: Width -> Height -> IO Board
generateInitialBoard width height = do
  let boardLines = init $ tail wallBoard 
  
  randomBoard <- replicateM (length wallBoard - 2) (
    do
      boardLine <- replicateM (length (head boardLines) - 2) untilValid'
      return $ Wall : boardLine ++ [Wall]
    )
    
  pure $ Board $ head wallBoard : randomBoard ++ [last wallBoard]
  where
    walllLine = replicate width Wall
    wallBoard = replicate height walllLine
    untilValid FiredUp = untilValid'
    untilValid FiredDown = untilValid'
    untilValid FiredLeft = untilValid'
    untilValid FiredRight = untilValid'
    untilValid Shadow = return Empty
    untilValid Unknow = return Empty
    untilValid Me =  return Empty
    untilValid x = return x
    untilValid' = untilValid =<< randomCell

randomCell :: IO Cell
randomCell = randomIO

nextAction :: Board -> BoardAction
nextAction (Board b) = BoardAction $ map (map (\case
                                                         Unknow -> NoOp
                                                         Wall ->  NoOp
                                                         Enemy -> MoveUp
                                                         Opponent -> MoveDown
                                                         Me -> MoveLeft
                                                         _ -> NoOp
                                              )
                                         ) b


move :: Board -> (Action, (Width, Height)) -> Board
move (Board b) (a, (x, y)) = case a of
                     MoveUp ->
                       let (oldRowBefore,_:oldRowAfter) = splitAt (x) col
                           (oldColBefore,col:oldColAfter) = splitAt (y) b
                       in Board $ oldColBefore ++ (oldRowBefore ++ (Empty : oldRowAfter)) : oldColAfter
                     MoveDown -> Board b
                     MoveRight -> Board b
                     MoveLeft -> Board b
                     _ -> Board b

applyAction :: Board -> BoardAction -> Board
applyAction (Board b) (BoardAction a) =
  Board $ zipWith (zipWith foo) b a
  where
    foo b a = case (b, a) of
                  (b, NoOp) -> b
                  (_, MoveUp) -> Empty
                  (_, MoveDown) -> Empty
                  (_, MoveRight) -> Empty
                  (_, MoveLeft) -> Empty
                  (b, FireUp) -> b
                  (b, FireDown) -> b
                  (b, FireLeft) -> b
                  (b, FireRight) -> b


nextRound :: Board -> IO Board
nextRound b =
  return b

boardX = 6
boardY = 5

someFunc :: IO ()
someFunc = do
  randomBoard <- generateInitialBoard boardX boardY
  print randomBoard
--   _ <- threadDelay 1000000
  
  let actions = nextAction randomBoard
  print actions
--  newBoard <- nextRound randomBoard
  -- map (move randomBoard)
  let (BoardAction act) = actions
  let foo = [ (x,y) | y <- [0..(boardY-1)], x <- [0..(boardX-1)]]
  print $ concat act
  print foo
  let actionWithCoord = zip (concat act) foo
  print actionWithCoord
  print $ foldr (\a b -> move b a) randomBoard actionWithCoord

  -- print $ applyAction randomBoard actions
