### Load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")


### ===============================================================================
# Base prediction of all deaths
# Adding survival column with all 0's for deaths
test$Survived <- 0

# Creating submission file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "alldeaths.csv", row.names = FALSE)

### =============================================================================== 
# Starting to explore explanatory variables
# Looking for relationship between Sex and survival
prop.table(table(train$Sex, train$Survived), 1)

# Clearly, females had a much higher chance of surviving.
# Let's alter our prediction to show this
test$Survived <- 0
test$Survived[test$Sex == "female"] <- 1

# Creating submission file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "femaleslive.csv", row.names = FALSE)

### Looking into age as a predictor
summary(train$Age)

# Adding a variable to identify children ( < 18 years old)
train$Child <- 0
train$Child[train$Age < 18] <- 1

# Looking at survival differences for child v adult
aggregate(Survived ~ Child + Sex, data = train, FUN = sum)

#Total number of people in these subsets
aggregate(Survived ~ Child + Sex, data = train, FUN = length)

# Looking at this proportion
aggregate(Survived ~ Child + Sex, data = train, FUN = function(x) (sum(x)/length(x)))

# Looks like male children were more likely to survive than male adults, but still
# not as likely as females of any sort

### Looking into social class variables
# Pclass just is 1st, 2nd or 3rd so don't really need to do anything to it

# Fare is continuous so we need to batch it up
train$Fare2 <- "30+"
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- "20 - 30"
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- "10 - 20"
train$Fare2[train$Fare < 10] <- "<10"

### Now looking at survival broken down by Sex, PClass and Fare2
aggregate(Survived ~ Fare2 + Pclass + Sex, data = train,
          FUN = function(x) (sum(x)/length(x)))

# There seem to be several interactions of note here--for now only focus on Pclass
# 3 females having low survival rates

# Submitting this insight
test$Survived <- 0
test$Survived[test$Sex == "female"] <- 1
test$Survived[test$Sex == "female" & test$Pclass == 3 & test$Fare >= 20] <- 0

submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, "Pclass3femalesdie.csv", row.names = FALSE)

### ===============================================================================
# Start of machine learning
library(rpart)
tree.fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                  data = train, method = "class" )

# examining our tree
library(rattle); library(rpart.plot); library(RColorBrewer)
fancyRpartPlot(tree.fit)

# Making prediction
tree.prediction <- predict(tree.fit, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = tree.prediction)
write.csv(submit, file = "tree1.csv", row.names = FALSE)

### ===============================================================================
# Little bit of feature engineering

# first let's merge the train and test so all features are included in both sets
train <- read.csv("train.csv")
test <- read.csv("test.csv")
test$Survived <- NA
combi <- rbind(train, test)

# casting names to character strings
combi$Name <- as.character(combi$Name)

# extracting the title from the name variable
combi$Title <- sapply(combi$Name, FUN = function(x) {strsplit(x, split = "[,.]")[[1]][2]})
# removing first space from Title
combi$Title <- sub(" ", "", combi$Title)

# combining similar titles which are infrequent
combi$Title[combi$Title %in% c("Mlle", "Mme")] <- "Mlle"
combi$Title[combi$Title %in% c("Capt", "Don", "Major", "Sir")] <- "Sir"
combi$Title[combi$Title %in% c("Dona", "Lady", "Jonkheer", "the Countess")] <- "Lady"

# converting back to factor categories
combi$Title <- factor(combi$Title)


# dealing with sibsp and parch variables
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# looking into specific families
combi$Surname <- sapply(combi$Name, FUN = function(x) {strsplit(x, split = "[,.]")[[1]][1]})
# combining surname and familysize to get id's for specific families
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep = "")

# Dealing with large families only by classifying those with <= 2 as just a small family
combi$FamilyID[combi$FamilySize <= 2] <- "Small"

# looking at FamilyID frequency to clean further
famIDs <- data.frame(table(combi$FamilyID))
# getting rid of any with frequency <= 2
famIDs <- famIDs[famIDs$Freq <= 2,]

# using famIDs dataframe to filter rest of small families into the small category
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- "Small"
combi$FamilyID <- factor(combi$FamilyID)

### splitting back into train and test sets
train <- combi[1:891,]
test <- combi[892:1309,]

# Creating new tree model including engineered features
tree.fit2 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
                   + Title + FamilySize + FamilyID, data = train, method = "class")

# creating data frame for submission
tree2.prediction <- predict(tree.fit2, test, type = "class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = tree2.prediction)
write.csv(submit, "engineeredfeatures1.csv", row.names = FALSE)


### ===============================================================================
# More complex modeling
# Going to use random forest--need to deal with missing values before running model
# age has about 20% missing values
summary(combi$Age)

# using a shallow tree to predict the ages instead of naive median/mean
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title +
                  FamilySize, data = combi[!is.na(combi$Age),], method = "anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

# checking for other missing values and replacing
summary(combi)

combi$Embarked[which(combi$Embarked == "")] = "S"
combi$Embarked <- factor(combi$Embarked)

combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm = TRUE)

# Random forest can only deal with factors with < 32 levels 
# Means we need to deal with FamilyID
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- "Small"
combi$FamilyID2 <- factor(combi$FamilyID2)

# splitting combi back to train and test sets
train <- combi[1:891,]
test <- combi[892:1309,]

### running randomForest model
library(randomForest)
set.seed(123)
# casting Survived as factor changes the model to classification instead of regression
# Importance allows you to inspect the variable importance
rf.fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch +
                         Fare + Embarked + Title + FamilySize + FamilyID2,
                       data = train, importance = TRUE, ntree = 2000)

# Looking into variable importance
varImpPlot(rf.fit)

# submitting first rf effort
rf.prediction <- predict(rf.fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = rf.prediction)
write.csv(submit, file = "rf1.csv", row.names = FALSE)

### ================================================================================
# conditional inference tree
library(party)
set.seed(415)
party.fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch +
                       Fare + Embarked + Title + FamilySize + FamilyID,
                     data = train, controls = cforest_unbiased(ntree = 2000, mtry = 3))
# conditional tree prediction
cforest.prediction <- predict(party.fit, test, OOB = TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = cforest.prediction)
write.csv(submit, file = "cforest1.csv", row.names = FALSE)


### ===============================================================================
# Exploring Cabin variable
# First get the non-missing Cabin information into two new variables--Deck and 
# Location
combi$Deck <- sapply(as.character(combi$Cabin),
                     FUN = function(x) {strsplit(x, split = "")[[1]][1]})
combi$Deck[which(is.na(combi$Deck))] <- "UNK"
combi$Deck <- factor(combi$Deck)
library(stringr)
library(plyr)
combi$cabin.last.digit <- str_sub(combi$Cabin, -1)
combi$Location <- "UNK"
combi$Location[which(combi$cabin.last.digit %in% c("0", "2", "4", "6", "8"))] <- "Port"
combi$Location[which(combi$cabin.last.digit %in% c("1", "3", "5", "7", "9"))] <- "Starboard"
combi$Location <- factor(combi$Location)
combi$cabin.last.digit <- NULL
