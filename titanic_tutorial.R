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
