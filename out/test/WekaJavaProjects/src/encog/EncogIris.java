package encog;

import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.util.csv.CSVFormat;

import java.io.File;

public class EncogIris {
    public static void main(String[] args) {
        File irisFile = new File("data/iris.data.csv");
        VersatileDataSource source = new CSVDataSource(irisFile, false, CSVFormat.DECIMAL_POINT);

        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
        data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
        data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
        data.defineSourceColumn("petal-width", 3, ColumnType.continuous);

        ColumnDefinition outputColumn = data.defineSourceColumn("species", 4, ColumnType.nominal);
        data.analyze();
    }
}
