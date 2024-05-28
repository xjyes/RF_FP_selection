library(readxl)
library(tidyr)
setwd("/Users/xujingyi/Documents/BGI/酶切假阳性预测/random forest run file")

result <- read_excel("result.xlsx")
result <- unite(result, gene_exon, c(Hugo_Symbol,EXON), remove = F)

true <- result[result$label == "True",]  # false positive
false <- result[result$label == "False",] # true positive

true <- as.data.frame(table(true$gene_exon))
false <- as.data.frame(table(false$gene_exon))

true <- true[order(-true$Freq),]
false <- false[order(-false$Freq),]

true <- separate(true,Var1,into = c("Gene", "Exon"),sep="_")
false <- separate(false,Var1,into = c("Gene", "Exon"),sep="_")

write.csv(true,file = "/Users/xujingyi/Documents/BGI/酶切假阳性预测/results/random forest/Gene_exon_true.csv",row.names = F)
write.csv(false,file = "/Users/xujingyi/Documents/BGI/酶切假阳性预测/results/random forest/Gene_exon_false.csv",row.names = F)
